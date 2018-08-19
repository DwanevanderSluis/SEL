

# To Push A file through an already trained model

sellibrary/main/main_send_data_through_model_trec_eval.py

# To produce the output (intermediate file)

#### SEL on dexter

sellibrary/main/main_build_sel_model.py

#### TF om dexter

sellibrary/main/main_use_or_build_baseline_tf_model.py

#### Sentiment on Dexter

sellibrary/main/main_build_sentiment_model.py

#### SEL on washington post

main_build_sel_model_for_washington_post.py



# AWS Batch

To run on signal batch servers, from root directory run
signal-experiment deploy -s . -m 12288 -c "/bin/bash ./run_on_signal.sh 1 1"

To start an instance and drop to the comand shell
docker run -it 2c920db4235a /bin/bash

To list docker images
docker images

To get the stats on containers
docker container stats

To list all docker containers
docker container ps

To remove unneeded docker images
docker images prune


To remove only in a certain state
docker ps -aq --no-trunc -f status=created | xargs docker rm

To edit docker image
vi Dockerfile

To copy files into the correct places for the docker image, run
./copy_files_here.sh
from project root/files_for_docker


# washington post

###### Build json file from 

wikipedia_for_washingtonpost/newsir18-entity-ranking-topics.xml



#### SEL on washington post

main_build_sel_model_for_washington_post.py

