"""
SiliconLife Eyeflow
Class to operate message queue server/client

Author: Alex Sobral de Freitas
"""

import os
import json
import datetime
import time
from typing import Dict
import pika
from bson.objectid import ObjectId

from eyeflow_sdk.log_obj import log, CONFIG
#----------------------------------------------------------------------------------------------------------------------------------

class QueueChannel(object):
    def __init__(self, queue, message_call, parent):
        self._queue = queue
        self._message_call = message_call
        self._parent = parent
        self._consumer_tag = None


    def on_channel_open(self, channel):
        log.info('Channel opened')
        self._channel = channel

        log.info('Adding channel close callback')
        self._channel.add_on_close_callback(self.on_channel_closed)

        log.info('Declaring queue %s' % self._queue)
        self._channel.queue_declare(queue=self._queue,
                                    callback=self.on_queue_declareok,
                                    durable=True)

        log.info('Adding process consumer cancellation callback')
        self._channel.add_on_cancel_callback(self.on_consumer_cancelled)


    def on_message(self, unused_channel, basic_deliver, properties, body):
        log.info('Received message # %s from %s: %s', basic_deliver.delivery_tag, properties.app_id, body)
        parms = json.loads(body.decode())
        self._message_call(parms)

        # Acknowledge the message delivery from RabbitMQ by sending a Basic.Ack RPC method for the delivery tag.
        self._channel.basic_ack(basic_deliver.delivery_tag)


    def on_queue_declareok(self, method_frame):
        self._consumer_tag = self._channel.basic_consume(queue=self._queue,
                                                         on_message_callback=self.on_message)


    def on_channel_closed(self, channel, reason):
        log.warning('Channel %i was closed: %s', channel, reason)
        self._parent._connection.close()


    def on_consumer_cancelled(self, method_frame):
        log.info('Consumer was cancelled remotely, shutting down: %r', method_frame)
        if self._channel:
            self._channel.close()


    def stop_consuming(self):
        if self._channel:
            log.info('Sending a Basic.Cancel RPC command to RabbitMQ')
            self._channel.basic_cancel(self.on_cancelok, self._consumer_tag)


    def on_cancelok(self, unused_frame):
        log.info('RabbitMQ acknowledged the cancellation of the consumer')
        log.info('Closing the channel')
        self._channel.close()
#----------------------------------------------------------------------------------------------------------------------------------

class TopicChannel(object):
    def __init__(self, topic, route, message_call, parent):
        self._topic = topic
        self._routing_key = route
        self._queue_name = None
        self._message_call = message_call
        self._parent = parent
        self._consumer_tag = None


    def on_channel_open(self, channel):
        log.info('Channel opened')
        self._channel = channel

        log.info('Adding channel close callback')
        self._channel.add_on_close_callback(self.on_channel_closed)

        log.info('Declaring exchange %s' % self._topic)
        self._channel.exchange_declare(exchange=self._topic,
                                       callback=self.on_exchange_declareok,
                                       durable=True,
                                       exchange_type='topic')


    def on_exchange_declareok(self, unused_frame):
        log.info('Exchange declared')
        log.info('Declaring queue')
        self._channel.queue_declare(queue='',
                                    callback=self.on_queue_declareok,
                                    exclusive=True,
                                    auto_delete=True)


    def on_queue_declareok(self, method_frame):
        log.info('Queue declare ok: consumer_count={}, message_count={}, queue={}'.format(method_frame.method.consumer_count, method_frame.method.message_count, method_frame.method.queue))
        self._queue_name = method_frame.method.queue

        log.info('Binding %s to %s with %s' % (self._topic, self._queue_name, self._routing_key))
        self._channel.queue_bind(exchange=self._topic,
                                 routing_key=self._routing_key,
                                 queue=self._queue_name,
                                 callback=self.on_bindok)


    def on_bindok(self, unused_frame):
        log.info('Queue bound')
        log.info('Adding process consumer cancellation callback')
        self._channel.add_on_cancel_callback(self.on_consumer_cancelled)
        self._consumer_tag = self._channel.basic_consume(queue=self._queue_name,
                                                         on_message_callback=self.on_message)


    def on_message(self, unused_channel, basic_deliver, properties, body):
        event_parms = json.loads(body.decode())
        queue_parms = {"timestamp": datetime.datetime.now().isoformat(),
                       "routing_key": basic_deliver.routing_key,
                       "app_id": properties.app_id
        }
        self._message_call(queue_parms, event_parms)

        # Acknowledge the message delivery from RabbitMQ by sending a Basic.Ack RPC method for the delivery tag.
        self._channel.basic_ack(basic_deliver.delivery_tag)


    def on_channel_closed(self, channel, reason):
        log.warning('Channel %i was closed: %s', channel, reason)
        self._parent._connection.close()


    def on_consumer_cancelled(self, method_frame):
        log.info('Consumer was cancelled remotely, shutting down: %r', method_frame)
        if self._channel:
            self._channel.close()


    def stop_consuming(self):
        if self._channel:
            log.info('Sending a Basic.Cancel RPC command to RabbitMQ')
            self._channel.basic_cancel(self.on_cancelok, self._consumer_tag)


    def on_cancelok(self, unused_frame):
        log.info('RabbitMQ acknowledged the cancellation of the consumer')
        log.info('Closing the channel')
        self._channel.close()
#----------------------------------------------------------------------------------------------------------------------------------

class QueueConsumer(object):
    def __init__(self, amqp_url, queues=None, topic=None):
        self._connection = None
        self._channels = []
        self._queues = queues
        self._topic = topic
        self._connection_tries = 0
        self._closing = False
        self._url = amqp_url


    def connect(self):
        log.info('Connecting to %s', self._url)
        self._connection = pika.SelectConnection(pika.URLParameters(self._url),
                                                 on_open_callback=self.on_connection_open,
                                                 on_open_error_callback=self.on_connection_error)


    def on_connection_error(self, unused_connection, excpt):
        if self._connection_tries < 5:
            log.warning('Fail to connect - {}. Wait 5 seconds'.format(repr(excpt)))
            time.sleep(5)
            self._connection_tries += 1
            self.connect()
            self._connection.ioloop.start()
        else:
            log.error('Fail to connect to broker')
            exit(1)


    def on_connection_open(self, unused_connection):
        log.info('Connection opened')
        log.info('Adding connection close callback')
        self._connection.add_on_close_callback(self.on_connection_closed)

        if self._queues is not None:
            for queue in self._queues:
                log.info('Creating queue channel: ' + queue[0])
                new_channel = QueueChannel(queue[0], queue[1], self)
                self._channels.append(new_channel)
                self._connection.channel(on_open_callback=new_channel.on_channel_open)
        elif self._topic is not None:
            log.info('Creating topic channel: ' + self._topic[0])
            new_channel = TopicChannel(self._topic[0], self._topic[1], self._topic[2], self)
            self._channels.append(new_channel)
            self._connection.channel(on_open_callback=new_channel.on_channel_open)


    def on_connection_closed(self, connection, reason):
        self._channels = []
        if self._closing:
            self._connection.ioloop.stop()
        else:
            log.warning('Connection closed, reopening in 5 seconds: %s', reason)
            self._connection.ioloop.call_later(5, self.reconnect)


    def reconnect(self):
        # This is the old connection IOLoop instance, stop its ioloop
        self._connection.ioloop.stop()

        if not self._closing:
            # Create a new connection
            self.connect()

            # There is now a new connection, needs a new ioloop to run
            self._connection.ioloop.start()


    def run(self):
        self._connection_tries = 0
        self.connect()
        self._connection.ioloop.start()


    def stop(self):
        log.info('Stopping')
        self._closing = True
        for channel in self._channels:
            channel.stop_consuming()

        self._connection.ioloop.stop()
        log.info('Stopped')


    def close_connection(self):
        log.info('Closing connection')
        self._connection.close()
#----------------------------------------------------------------------------------------------------------------------------------

def publish_message(message, queue, topic='') -> None:
    """
    Publish a message to a queue or a exchange
    message: dict - Dict with field-value of the message
    queue: str - Name of the queue to publish the message
    topic: str - Topic to publish, leave blank to publish to a queue
    """
    if "MQ_URL" in os.environ:
        mq_url = os.environ["MQ_URL"]
    elif "mq-service" in CONFIG:
        mq_parms = CONFIG["mq-service"]
        mq_url = f"amqp://{mq_parms['user']}:{mq_parms['pass']}@{mq_parms['host']}:{mq_parms['port']}"
    else:
        raise Exception('Need parms of connection os.environ["MQ_URL"] or CONFIG["mq-service"]')

    mq_connection = pika.BlockingConnection(pika.URLParameters(mq_url))
    mq_channel = mq_connection.channel()

    mq_channel.basic_publish(
        exchange=topic,
        routing_key=queue,
        properties=pika.BasicProperties(content_type="application/json"),
        body=json.dumps(message, default=str)
    )

    mq_channel.close()
    mq_connection.close()
#----------------------------------------------------------------------------------------------------------------------------------


def send_message(message, queue, topic='', timeout=5) -> Dict:
    """
    Send a message to a queue or a exchange and return response to caller
    message: dict - Dict with field-value of the message
    queue: str - Name of the queue to publish the message
    topic: str - Topic to publish, leave blank to publish to a queue
    timeout: int - Timeout in seconds
    """

    if "MQ_URL" in os.environ:
        mq_url = os.environ["MQ_URL"]
    elif "mq-service" in CONFIG:
        mq_parms = CONFIG["mq-service"]
        mq_url = f"amqp://{mq_parms['user']}:{mq_parms['pass']}@{mq_parms['host']}:{mq_parms['port']}"
    else:
        raise Exception('Need parms of connection os.environ["MQ_URL"] or CONFIG["mq-service"]')

    mq_connection = pika.BlockingConnection(pika.URLParameters(mq_url))
    mq_channel = mq_connection.channel()

    corr_id = None
    response = None

    def on_response(channel, basic_deliver, properties, body):
        nonlocal response
        if corr_id == properties.correlation_id:
            response = json.loads(body.decode())

    result = mq_channel.queue_declare(queue='', exclusive=True)
    response_queue_name = result.method.queue

    mq_channel.basic_consume(
        on_message_callback=on_response,
        queue=response_queue_name,
        auto_ack=True
    )

    corr_id = str(ObjectId())

    mq_channel.basic_publish(
        exchange=topic,
        routing_key=queue,
        properties=pika.BasicProperties(
            content_type="application/json",
            reply_to=response_queue_name,
            correlation_id=corr_id,
        ),
        body=json.dumps(message, default=str)
    )

    start_timeout = datetime.datetime.now()
    while not response:
        mq_connection.process_data_events()
        time.sleep(0.01)
        if (datetime.datetime.now() - start_timeout).total_seconds() > timeout:
            raise Exception("Timeout waiting response")

    return response
#----------------------------------------------------------------------------------------------------------------------------------
