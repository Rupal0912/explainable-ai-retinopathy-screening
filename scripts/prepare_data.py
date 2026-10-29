from pathlib import Path
import hashlib,json,pandas as pd
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def main():
 data=ROOT/"data"; images=data/"train_images"; source=data/"train.csv"; out=data/"splits"
 if not source.exists():raise SystemExit("Place APTOS train.csv at data/train.csv.")
 df=pd.read_csv(source)
 if not {"id_code","diagnosis"}.issubset(df.columns):raise SystemExit("train.csv needs id_code and diagnosis columns.")
 df["diagnosis"]=pd.to_numeric(df["diagnosis"],errors="coerce");df=df.dropna(subset=["diagnosis"]);df["diagnosis"]=df["diagnosis"].astype(int)
 if not df.diagnosis.between(0,4).all():raise SystemExit("Diagnosis values must be 0..4.")
 seen=set();keep=[]
 for id in df.id_code:
  p=images/f"{id}.png"
  if not p.exists():raise SystemExit(f"Missing image: {p}")
  h=sha(p)
  if h not in seen:seen.add(h);keep.append(id)
 df=df[df.id_code.isin(keep)]
 trv,te=train_test_split(df,test_size=.15,random_state=42,stratify=df.diagnosis)
 tr,va=train_test_split(trv,test_size=.15/.85,random_state=42,stratify=trv.diagnosis)
 out.mkdir(parents=True,exist_ok=True)
 tr.to_csv(out/"train.csv",index=False);va.to_csv(out/"val.csv",index=False);te.to_csv(out/"test.csv",index=False)
 report={"counts":{"train":len(tr),"val":len(va),"test":len(te)},"seed":42,"test_used_for_selection":False,
 "class_distribution":{k:v.diagnosis.value_counts().sort_index().to_dict() for k,v in [("train",tr),("val",va),("test",te)]}}
 (out/"split_report.json").write_text(json.dumps(report,indent=2),encoding="utf8");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
