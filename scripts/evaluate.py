from pathlib import Path
import json,sys,platform,datetime
import numpy as np,pandas as pd,torch
from PIL import Image
from sklearn.metrics import accuracy_score,f1_score,cohen_kappa_score,confusion_matrix,precision_recall_fscore_support
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app_core import build_model,preprocess,LABELS
def main():
 out=ROOT/"models"/"test_metrics.json"
 if out.exists():raise SystemExit("test_metrics.json exists; refusing to re-evaluate held-out test data. Remove it only after a documented decision to conduct a new evaluation.")
 csv=ROOT/"data"/"splits"/"test.csv"; ck=ROOT/"models"/"best.pt"
 if not csv.exists() or not ck.exists():raise SystemExit("Need test split and models/best.pt first.")
 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu");m=build_model(False).to(dev);saved=torch.load(ck,map_location=dev,weights_only=False);m.load_state_dict(saved["model_state_dict"]);m.eval()
 df=pd.read_csv(csv);ys=[];ps=[]
 with torch.inference_mode():
  for _,r in df.iterrows():
   im=Image.open(ROOT/"data"/"train_images"/f"{r.id_code}.png").convert("RGB")
   # Use same inference preprocessing implementation.
   from app_core import preprocess
   x=preprocess(im).unsqueeze(0).to(dev);ps.append(int(m(x).argmax(1).item()));ys.append(int(r.diagnosis))
 p,r,f,s=precision_recall_fscore_support(ys,ps,labels=[0,1,2,3,4],zero_division=0)
 t=np.array(ys)>=2;q=np.array(ps)>=2;tp=int((t&q).sum());tn=int((~t&~q).sum());fp=int((~t&q).sum());fn=int((t&~q).sum())
 result={"educational_only":True,"device":str(dev),"evaluation_date_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "checkpoint":str(ck.relative_to(ROOT)),"n_test":len(ys),"test_class_counts":{LABELS[i]:int(np.sum(np.array(ys)==i)) for i in range(5)},
 "accuracy":float(accuracy_score(ys,ps)),
 "macro_f1":float(f1_score(ys,ps,average="macro",zero_division=0)),"quadratic_weighted_kappa":float(cohen_kappa_score(ys,ps,weights="quadratic")),
 "per_class":{LABELS[i]:{"precision":float(p[i]),"recall":float(r[i]),"f1":float(f[i]),"support":int(s[i])} for i in range(5)},
 "confusion_matrix":confusion_matrix(ys,ps,labels=[0,1,2,3,4]).tolist(),
 "grade_ge_2_demo_convention":{"sensitivity":tp/max(1,tp+fn),"specificity":tn/max(1,tn+fp)},
 "note":"Grade >=2 is a project convention, not a clinical threshold or guidance.",
 "limitations":"Single public dataset; no external or clinical validation. Confidence is not calibrated."}
 out.write_text(json.dumps(result,indent=2),encoding="utf8");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
