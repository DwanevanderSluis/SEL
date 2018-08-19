import boto3
import botocore
from sellibrary.locations import FileLocations


s3_client = boto3.client('s3')

# Call S3 to list current buckets
response = s3_client.list_buckets()

# Get a list of all bucket names from the response
buckets = [bucket['Name'] for bucket in response['Buckets']]
# Print out the bucket list
print("Bucket List: %s" % buckets)

BUCKET_NAME = 'entity-salience.rnd.signal' # replace with your bucket name
KEY = 'my_image_in_s3.jpg' # replace with your object key
path = FileLocations.get_dropbox_intermediate_path() + 'wp'

s3 = boto3.resource('s3')
try:
    bucket = s3.Bucket(BUCKET_NAME)
    for object in bucket.objects.all():
        print(object.key)
        # d = object.get()
        if object.key.find('sel_all_features_golden_spotter.washington_post.docnum') > -1:
            i = object.key.rfind('/')
            name = path+object.key[i:]
            print(name)
            bucket.download_file(object.key, name)


except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("The object does not exist.")
    else:
        raise


