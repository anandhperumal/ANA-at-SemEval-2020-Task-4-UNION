# ANA-at-SemEval-2020-Task-4-mUlti-task-learNIng-for-cOmmonsense-reasoNing-UNION
 

To Run the module as docker 
sudo docker build -t common_sense:0.0.1 .

sudo docker run -d -p 8080:8080 common_sense:0.0.1

To run it as a flask app : run the main.py file

To train a model form Scratch : MTD-NCH.py 

@inproceedings{konar-etal-2020-ana,

    title = "{ANA} at {S}em{E}val-2020 Task 4: {MU}lti-task lear{NI}ng for c{O}mmonsense reaso{N}ing ({UNION})",
    
    author = "Konar, Anandh  and
      
      Huang, Chenyang  and
      
      Trabelsi, Amine  and
      
      Zaiane, Osmar",
    
    booktitle = "Proceedings of the Fourteenth Workshop on Semantic Evaluation",
    month = dec,
    year = "2020",
    address = "Barcelona (online)",
    publisher = "International Committee for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.semeval-1.45",
    pages = "367--373",
    abstract = "In this paper, we describe our mUlti-task learNIng for cOmmonsense reasoNing (UNION) system submitted for Task C of the SemEval2020 Task 4, which is to generate a reason explaining why a given false statement is non-sensical. However, we found in the early experiments that simple adaptations such as fine-tuning GPT2 often yield dull and non-informative generations (e.g. simple negations). In order to generate more meaningful explanations, we propose UNION, a unified end-to-end framework, to utilize several existing commonsense datasets so that it allows a model to learn more dynamics under the scope of commonsense reasoning. In order to perform model selection efficiently, accurately, and promptly, we also propose a couple of auxiliary automatic evaluation metrics so that we can extensively compare the models from different perspectives. Our submitted system not only results in a good performance in the proposed metrics but also outperforms its competitors with the highest achieved score of 2.10 for human evaluation while remaining a BLEU score of 15.7. Our code is made publicly available.",
}
