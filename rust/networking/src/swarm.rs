//! Compat shim for the old libp2p code

use std::collections::HashMap;
use std::pin::Pin;

use futures_lite::Stream;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use zenoh::Result;
use zenoh::Session;
use zenoh::handlers::FifoChannelHandler;
use zenoh::liveliness::LivelinessToken;
use zenoh::pubsub::Publisher;
use zenoh::pubsub::Subscriber;
use zenoh::qos::CongestionControl;
use zenoh::sample::Sample;
use zenoh::sample::SampleKind;

#[derive(Debug)]
pub enum ToSwarm {
    Unsubscribe {
        topic: String,
        result_sender: oneshot::Sender<bool>,
    },
    Subscribe {
        topic: String,
        result_sender: oneshot::Sender<Result<bool>>,
    },
    Publish {
        topic: String,
        data: Vec<u8>,
        result_sender: oneshot::Sender<Result<()>>,
    },
}
#[derive(Debug)]
pub enum FromSwarm {
    Message { topic: String, data: Vec<u8> },
    Discovered {},
    Expired {},
}

pub type Topics = HashMap<String, (Subscriber<()>, Publisher<'static>)>;
pub struct Swarm {
    pub session: crate::Session,
    pub from_client: mpsc::Receiver<ToSwarm>,
}

fn namespace_key(namespace: &str) -> String {
    blake3::hash(namespace.as_bytes()).to_hex()[..16].to_owned()
}

fn namespaced_topic_key(namespace: &str, topic: &str) -> String {
    format!("namespaces/{}/topics/{topic}", namespace_key(namespace))
}

fn namespaced_liveliness_prefix(namespace: &str) -> String {
    format!("namespaces/{}/live", namespace_key(namespace))
}

fn namespaced_liveliness_key(namespace: &str, zid: &str) -> String {
    format!("{}/{zid}", namespaced_liveliness_prefix(namespace))
}

impl Swarm {
    pub fn into_stream(self) -> Pin<Box<dyn Stream<Item = FromSwarm> + Send>> {
        let Swarm {
            session,
            mut from_client,
        } = self;
        let namespace = session.namespace.clone();
        let stream = async_stream::stream! {
            let mut session = session;
            let (mut to_topics, mut from_topics) = mpsc::channel(1024);
            let mut topics = Topics::new();
            let Ok((_token, discovery)) = register_liveness(&mut session.z, &namespace).await else { return; };
            let liveliness_prefix = format!("{}/", namespaced_liveliness_prefix(&namespace));
            loop {
                tokio::select! {
                    msg = from_client.recv() => {
                        let Some(msg) = msg else { break };
                        on_message(
                            &mut session.z,
                            &namespace,
                            &mut topics,
                            &mut to_topics,
                            msg,
                        ).await;
                    }
                    event = from_topics.recv() => {
                        if let Some(event) = event {
                            yield event
                        }
                    }
                    token = discovery.recv_async() => {
                        if let Ok(token) = token {
                            let key_expr = token.key_expr().as_str().to_owned();
                            let zid = key_expr.strip_prefix(&liveliness_prefix);
                            yield match token.kind() {
                                SampleKind::Put => {
                                    log::info!("discovered: {zid:?}");
                                    FromSwarm::Discovered {}
                                }
                                SampleKind::Delete => {
                                    log::info!("expired: {zid:?}");
                                    FromSwarm::Expired {}
                                }
                            }
                        }

                    }
                }
            }
        };
        Box::pin(stream)
    }
}

async fn register_liveness(
    session: &mut Session,
    namespace: &str,
) -> Result<(LivelinessToken, Subscriber<FifoChannelHandler<Sample>>)> {
    let liveliness_prefix = namespaced_liveliness_prefix(namespace);
    let token = session
        .liveliness()
        .declare_token(namespaced_liveliness_key(
            namespace,
            &session.zid().to_string(),
        ))
        .await?;
    let sub = session
        .liveliness()
        .declare_subscriber(format!("{liveliness_prefix}/*"))
        .history(true)
        .await?;
    Ok((token, sub))
}

async fn on_message(
    session: &mut Session,
    namespace: &str,
    topics: &mut Topics,
    to_topics: &mut mpsc::Sender<FromSwarm>,
    msg: ToSwarm,
) {
    match msg {
        ToSwarm::Publish {
            topic,
            data,
            result_sender,
        } => {
            let res = match topics.get(&topic) {
                Some(topic) => topic.1.put(data).await,
                None => {
                    // TODO: this should be an error but the python FromSwarm is somewhat nondeterministic
                    Ok(()) //Err("not subscribed to topic!".into()),
                }
            };
            _ = result_sender.send(res);
        }
        ToSwarm::Unsubscribe {
            topic,
            result_sender,
        } => {
            let Some((_, (subscriber, publisher))) = topics.remove_entry(&topic) else {
                _ = result_sender.send(false);
                return;
            };
            _ = publisher.undeclare().await;
            _ = subscriber.undeclare().await;
            _ = result_sender.send(true);
        }
        ToSwarm::Subscribe {
            topic,
            result_sender,
        } => {
            assert!(topic.is_ascii());
            if topics.contains_key(&topic) {
                _ = result_sender.send(Ok(false));
                return;
            }

            let topic_key = namespaced_topic_key(namespace, &topic);
            let publisher_res = session
                .declare_publisher(topic_key.clone())
                .congestion_control(CongestionControl::Block)
                .await;
            let publisher = match publisher_res {
                Ok(p) => p,
                Err(e) => {
                    _ = result_sender.send(Err(e));
                    return;
                }
            };

            let subscriber_res = session
                .declare_subscriber(topic_key)
                .allowed_origin(zenoh::sample::Locality::Remote)
                .callback({
                    let sender = to_topics.clone();
                    let topic = topic.clone();
                    move |sample| {
                        if sample.kind() != SampleKind::Put {
                            return;
                        }
                        _ = sender.try_send(FromSwarm::Message {
                            topic: topic.clone(),
                            data: sample.payload().to_bytes().to_vec(),
                        });
                    }
                })
                .await;
            let subscriber = match subscriber_res {
                Ok(s) => s,
                Err(e) => {
                    _ = result_sender.send(Err(e));
                    return;
                }
            };

            assert!(topics.insert(topic, (subscriber, publisher)).is_none());
            _ = result_sender.send(Ok(true));
        }
    }
}

pub async fn create_swarm(
    identity: &str,
    namespace: &str,
    from_client: mpsc::Receiver<ToSwarm>,
    listen_port: u16,
    discovery_service_port: u16,
    bootstrap_endpoints: &[String],
) -> Result<Swarm> {
    let cfg = crate::cfg(identity, listen_port, bootstrap_endpoints)?;
    let session = crate::open(cfg, namespace, listen_port, discovery_service_port).await?;
    Ok(Swarm {
        session,
        from_client,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_lite::StreamExt as _;
    use std::net::{TcpListener, UdpSocket};
    use std::time::Duration;
    use tokio::task::JoinHandle;
    use tokio::time::timeout;

    fn free_tcp_port() -> u16 {
        TcpListener::bind("[::1]:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    fn free_udp_port() -> u16 {
        UdpSocket::bind("[::1]:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    async fn start_swarm(
        identity: &str,
        namespace: &str,
        listen_port: u16,
        bootstrap_endpoints: &[String],
    ) -> (
        mpsc::Sender<ToSwarm>,
        mpsc::Receiver<FromSwarm>,
        JoinHandle<()>,
    ) {
        let (to_swarm, from_client) = mpsc::channel(8);
        let swarm = create_swarm(
            identity,
            namespace,
            from_client,
            listen_port,
            free_udp_port(),
            bootstrap_endpoints,
        )
        .await
        .unwrap();
        let mut stream = swarm.into_stream();
        let (events_tx, events_rx) = mpsc::channel(16);
        let task = tokio::spawn(async move {
            while let Some(event) = stream.next().await {
                if events_tx.send(event).await.is_err() {
                    break;
                }
            }
        });
        (to_swarm, events_rx, task)
    }

    async fn subscribe(to_swarm: &mpsc::Sender<ToSwarm>, topic: &str) {
        let (result_sender, result_receiver) = oneshot::channel();
        to_swarm
            .send(ToSwarm::Subscribe {
                topic: topic.to_owned(),
                result_sender,
            })
            .await
            .unwrap();
        assert!(result_receiver.await.unwrap().unwrap());
    }

    async fn publish(to_swarm: &mpsc::Sender<ToSwarm>, topic: &str, data: &[u8]) {
        let (result_sender, result_receiver) = oneshot::channel();
        to_swarm
            .send(ToSwarm::Publish {
                topic: topic.to_owned(),
                data: data.to_vec(),
                result_sender,
            })
            .await
            .unwrap();
        result_receiver.await.unwrap().unwrap();
    }

    async fn wait_for_discovered(events: &mut mpsc::Receiver<FromSwarm>) {
        timeout(Duration::from_secs(5), async {
            while let Some(event) = events.recv().await {
                if matches!(event, FromSwarm::Discovered {}) {
                    return;
                }
            }
            panic!("swarm event stream closed before discovery");
        })
        .await
        .expect("same-namespace peers did not discover each other");
    }

    async fn wait_for_message(
        events: &mut mpsc::Receiver<FromSwarm>,
        topic: &str,
    ) -> Option<Vec<u8>> {
        while let Some(event) = events.recv().await {
            if let FromSwarm::Message {
                topic: received_topic,
                data,
            } = event
                && received_topic == topic
            {
                return Some(data);
            }
        }
        None
    }

    #[test]
    fn namespace_scopes_topics_and_liveliness() {
        assert_ne!(
            namespaced_topic_key("cluster-a", "global_events"),
            namespaced_topic_key("cluster-b", "global_events")
        );
        assert_ne!(
            namespaced_liveliness_key("cluster-a", "abc"),
            namespaced_liveliness_key("cluster-b", "abc")
        );
        assert_eq!(
            namespaced_topic_key("cluster-a", "global_events"),
            namespaced_topic_key("cluster-a", "global_events")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn bootstrap_exchanges_only_same_namespace_traffic() {
        let topic = "global_events";
        let primary_port = free_tcp_port();
        let primary_endpoint = vec![format!("tcp/[::1]:{primary_port}")];
        let (primary_tx, mut primary_events, primary_task) =
            start_swarm("1", "cluster-a", primary_port, &[]).await;
        let (same_tx, mut same_events, same_task) =
            start_swarm("2", "cluster-a", free_tcp_port(), &primary_endpoint).await;
        let (other_tx, mut other_events, other_task) =
            start_swarm("3", "cluster-b", free_tcp_port(), &primary_endpoint).await;

        subscribe(&primary_tx, topic).await;
        subscribe(&same_tx, topic).await;
        subscribe(&other_tx, topic).await;
        wait_for_discovered(&mut primary_events).await;
        wait_for_discovered(&mut same_events).await;

        publish(&primary_tx, topic, b"same namespace").await;
        assert_eq!(
            timeout(
                Duration::from_secs(2),
                wait_for_message(&mut same_events, topic)
            )
            .await
            .unwrap(),
            Some(b"same namespace".to_vec())
        );
        assert!(
            timeout(
                Duration::from_millis(500),
                wait_for_message(&mut other_events, topic)
            )
            .await
            .is_err(),
            "different-namespace peer received EXO traffic"
        );

        publish(&other_tx, topic, b"other namespace").await;
        assert!(
            timeout(
                Duration::from_millis(500),
                wait_for_message(&mut primary_events, topic)
            )
            .await
            .is_err(),
            "primary received different-namespace EXO traffic"
        );

        primary_task.abort();
        same_task.abort();
        other_task.abort();
    }
}
