PythonAPI/pycocotools/cocoeval.py


- evaluate(self):
    - evaluateImg = self.evaluateImg -- calculates at each iou treshold the TP-s, FP-s and FN-s
        - TP 
        - FP - detection of something that is not
        - FN - not detected element
- cocoEval.accumulate()  # Aggregate results
    - Calculates at each iou 

- cocoEval.summarize()   # Display metrics


Iga IoU jaoks 

1. sorteerin predicted maskid (maski prediction )confidence järgi descending järjekorras
2. rida rea haaval lähen mask korraga ja 
    1. arvutan
        1. recall=TP/(TP+FN)
            1. (TP+FN) jääb alati konstantseks - see on kõigi olemasolevate ground truthide arve 
        2. precision=TP(TP+FP) 
    2. mida suuremaks lähen recall, seda väiksemaks läheb precision, sest suurema recalliga me püüame kinni kõiki ground truthid
3. inerpoleerin recall ja precision väärtuste pealt 0.1 vahedega 11 punktid ja leian AP
    1. $\frac{1}{11}\sum{\text{Precision at given Recall}}$