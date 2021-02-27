import numpy as np
import torch
from .parsing import args

leads = np.array(["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"])
lead2idx = dict(zip(leads, range(len(leads))))
num_leads = len(leads)

classes = np.array(sorted(("270492004", "164889003", "164890007", "426627000", 
                           "713427006", "713426002", "445118002", "39732003" , 
                           "164909002", "251146004", "698252002", "10370003" , 
                           "284470004", "427172004", "164947007", "111975006", 
                           "164917005", "47665007" , "427393009", "426177001", 
                           "426783006", "427084000", "164934002", "59931005"  )))
num_classes = len(classes)
equiv_class_groups = {
  "713427006": ["59118001"],
  "284470004": ["63593006"],
  "427172004": ["17338001"]
}

rfreq = 500
rtime = args.crop_time
rL = rfreq * rtime

class_labels = {
  "270492004": "1st degree AV block",
  "164889003": "Atrial fibrillation",
  "164890007": "Atrial flutter",
  "426627000": "Bradycardia",
  "713427006": "Complete right bundle branch block",
  "713426002": "Incomplete right bundle branch block",
  "445118002": "Left anterior fascicular block",
  "39732003" : "Left axis deviation",
  "164909002": "Left bundle branch block",
  "251146004": "Low QRS voltages",
  "698252002": "Nonspecific intraventricular conduction disorder",
  "10370003" : "Pacing rhythm",
  "284470004": "Premature atrial contraction",
  "427172004": "Premature ventricular contractions",
  "164947007": "Prolonged PR interval",
  "111975006": "Prolonged QT interval",
  "164917005": "Q wave abnormal",
  "47665007" : "Right axis deviation",
  "59118001" : "Right bundle branch block",
  "427393009": "Sinus arrhythmia",
  "426177001": "Sinus bradycardia",
  "426783006": "Sinus rhythm",
  "427084000": "Sinus tachycardia",
  "63593006" : "Supraventricular premature beats",
  "164934002": "T wave abnormal",
  "59931005" : "T wave inversion",
  "17338001" : "Ventricular premature beats"
}