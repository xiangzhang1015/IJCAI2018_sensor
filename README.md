# IJCAI2018_sensor

## Title: Multi-modality sensor data classification with selective attention

**PDF: [IJCAI2018](https://www.ijcai.org/proceedings/2018/0432.pdf), [arXiv](https://arxiv.org/abs/1804.05493)**

**Authors: [Xiang Zhang](http://xiangzhang.info/) (xiang_zhang@hms.harvard.edu), [Lina Yao](https://www.linayao.com/) (lina.yao@unsw.edu.au), Chaoran Huang, Sen Wang, Mingkui Tan, Guodong Long, Can Wang**

## Overview
This repository contains reproducible codes for the proposed model.  
In this paper, we present a robust and efficient multi-modality sensor data classification framework which integrates selective attention mechanism, deep reinforcement learning, and WAS-LSTM classification. In order to boost the chance of inter-dimension dependency in sensor features, we replicate and shuffle the sensor data. Additionally, the optimal spatial dependency is required for high-quality classification, for which we introduce the focal zone with attention mechanism. Furthermore, we extended the LSTM to exploit the crossrelationship among spatial dimensions, which is called WASLSTM, for classification. For more details please check our paper.


## Code
Python 2.7 TensorFlow 1.0

The env_spatial_hmm.py, RL_brain.py, and run_this.py are the reinforcement learning environment, client, and main codes. The RNN_attentionbar.py is the proposed WAS-LSTM classifier. More details are explained in the codes.


## Citing
If you find our work useful for your research, please consider citing this paper:

    @article{zhang2018multi,
      title={Multi-modality sensor data classification with selective attention},
      author={Zhang, Xiang and Yao, Lina and Huang, Chaoran and Wang, Sen and Tan, Mingkui and Long, Guodong and Wang, Can},
      journal={arXiv preprint arXiv:1804.05493},
      year={2018}
    }
    

## Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <xiang.alan.zhang@gmail.com>.


## License

This repository is licensed under the MIT License.
