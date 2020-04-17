# Neural Machine Translation 

Pytorch implementations of [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) for English to Hindi translation on [IITB En-Hi parallel corpus](http://www.cfilt.iitb.ac.in/iitb_parallel/).

## Prerequisites
- numpy==1.17.2
- torch==1.2.0
- torchtext==0.5.0
- tqdm==4.44.1
- indic-nlp-library==0.6
- Python 3.6+
 
## Usage

Install prerequisites with:
    
    pip3 install -r requirements.txt
    
    Download data from [IITB En-Hi parallel corpus](http://www.cfilt.iitb.ac.in/iitb_parallel/) and extract in Data folder.

To train model :
    
    python3 main.py  

To train with different data use \`!\`!\` as separator for source and target language data paths. For example:

    python3 main.py --training_data './Data/dev_test/test.en`!`!`./Data/dev_test/test.hi' --validation_data './Data/dev_test/dev.en`!`!`./Data/dev_test/dev.hi' 
    
To run in inference mode, provide trained model and dictionary paths. For example:

    python3 main.py --mode infer --load_model_path './trained_models/test_model.pt' --load_dic_path './trained_models/test_dic.pkl'

More configurations can be found [here](config.py).

## Reference

- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473).
- [IITB En-Hi parallel corpus](http://www.cfilt.iitb.ac.in/iitb_parallel/).
- [Seq2Seq tutorial](https://github.com/bentrevett/pytorch-seq2seq)


## Author

Shikhar / [@Shikhar](https://shikhar-s.github.io)
