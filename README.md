There are a lot of different datasets that have been used, and to keep it simple it is best that these are already avaliable. Especially the real world data set as it does a speedtest every time that it needs the throughput data. This essentially a controlled stress test, and should not be used too frequently, to not be disturbence for others on the internet and/or local network.

If the models folder do not contain the models, the train_model file will generate the models.
One MLP, one LR and one RF model. 

The xai_ig is used only by the mlp. 

xai lime and shap can be used by both LR and RF. 

PFI uses RF

gen real data is used to gather real data. Only works for now on linux debian enviorments. 

compare model is for quickly checking results of real data and/or cheking the performance of the model based on test sets.

eval driver recovery depend on having run xai_shap, xai_lime and xai ig or commenting out the scores in the file. This checks the driver recovery.


To get started, we need to create a .venv and download the required files from requierments.txt file.

Exact commands for running:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


python src/gen_synthetic . py
python src/baseline_detect . py

And so on... 


The datasets

Two are gatherd via gen_real and the ones called synthetic_network_system* are made with gen_synthetic.py. The ones with detections in the name are made after going through the baseline detector. 

Some pictures are also made. The called after the features are made from baseline detect. The other are made from the xAI. 