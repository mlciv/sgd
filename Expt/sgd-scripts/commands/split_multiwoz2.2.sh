#!/bin/bash

./multiwoz_splitter_good.sh /Expt/schema-guided/data/MultiWOZ_2.2/ /Expt/schema-guided/data/MultiWOZ_2.2_good/
./multiwoz_splitter.sh /Expt/schema-guided/data/MultiWOZ_2.2/ /Expt/schema-guided/data/MultiWOZ_2.2_hotel/ hotel
./multiwoz_splitter.sh /Expt/schema-guided/data/MultiWOZ_2.2/ /Expt/schema-guided/data/MultiWOZ_2.2_attraction/ attraction
./multiwoz_splitter.sh /Expt/schema-guided/data/MultiWOZ_2.2/ /Expt/schema-guided/data/MultiWOZ_2.2_restaurant/ restaurant
./multiwoz_splitter.sh /Expt/schema-guided/data/MultiWOZ_2.2/ /Expt/schema-guided/data/MultiWOZ_2.2_taxi/ taxi
./multiwoz_splitter.sh /Expt/schema-guided/data/MultiWOZ_2.2/ /Expt/schema-guided/data/MultiWOZ_2.2_train/ train

