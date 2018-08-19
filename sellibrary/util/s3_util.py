import boto3
import logging

class AWSUtil:
    # set up logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def copy_file_to_s3(self, filename):
        filename = filename.replace('//','/')
        self.logger.info('Copying %s to s3 ', filename)
        data = open(filename, "rb").read()
        self.logger.info('%d bytes read ', len(data))
        s3 = boto3.resource('s3')
        i = filename.rfind('/')
        filename2 = filename[(i+1):]
        object = s3.Object('entity-salience.rnd.signal', 'data/'+filename2)
        response = object.put(Body=data)
        self.logger.info('Copyied to %s ', 's3://entity-salience.rnd.signal/data/'+filename2)
        self.logger.info('Response %s ', response )


if __name__ == "__main__":
    app = AWSUtil()
    app.copy_file_to_s3('s3_util.py')