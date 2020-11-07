from eyeflow_sdk import message_queue
from eyeflow_sdk.log_obj import CONFIG

def call_trainer(parms):
    log.info("Trainer Received {}".format(parms))

if __name__ == '__main__':
    mq_parms = CONFIG["mq-service"]
    mq_url = f"amqp://{mq_parms['user']}:{mq_parms['pass']}@{mq_parms['host']}:{mq_parms['port']}"
    queues = [
        ('training_request', call_trainer)
    ]
    consumer = message_queue.QueueConsumer(mq_url, queues)
    consumer.run()
