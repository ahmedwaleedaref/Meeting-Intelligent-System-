from smi.labels import LABELS, TARGET_LABELS

def prf_from_sets(pred_ids: set[str], gold_ids: set[str]) -> dict:
    """
    P / R / F1 / support for one class.
    pred_ids : ids the model labelled with this class
    gold_ids : ids whose gold label is this class

    returns {"p": float, "r": float, "f1": float, "support": int}
    """
    number_of_segments_predicted_correctly : float = 0 
    for id in pred_ids : 
      if id in gold_ids : 
        number_of_segments_predicted_correctly +=1
        
    P =  number_of_segments_predicted_correctly / len(pred_ids) if len(pred_ids) > 0 else 0.0
    
    number_of_gold_segments_predicted_correctly : float = 0 
    for id in gold_ids : 
      if id in pred_ids : 
        number_of_gold_segments_predicted_correctly +=1
        
    R =  number_of_gold_segments_predicted_correctly / len(gold_ids) if len(gold_ids) > 0 else 0.0
    
    f1 = 2 * (P * R) / (P + R) if (P + R) > 0 else 0.0
    
    return {
      "p": P,
      "r": R,
      "f1": f1,
      "support": len(gold_ids)
    }


def per_class_metrics(records: dict[str, dict], gold: dict[str, str]) -> dict[str, dict]:
    """
    P / R / F1 / support for all 7 classes.
    records : output of load_predictions, seg_id -> {"label": ..., "scores": ...}
    gold    : seg_id -> gold label

    must raise ValueError if set(records) != set(gold)
    for each class in LABELS : build the two id sets and call prf_from_sets

    returns {label: {"p", "r", "f1", "support"}} with keys in LABELS order
    """
    if set(records) != set(gold) :
      raise ValueError("predictions and gold do not have the same seg_ids")

    metrics :  dict[str, dict] = {}
    for label in LABELS :
      pred_ids : set[str] = set()
      gold_ids : set[str] = set()
      for key , Value in records.items() :
        #Value it self is dict {}
        if Value["label"] == label :
          pred_ids.add(key)
      for key , Value in gold.items() :
          #Value here is the gold label string
          if Value == label :
            gold_ids.add(key)
      class_metrics = prf_from_sets(pred_ids , gold_ids)
      metrics[label] = class_metrics
    return metrics 

def macro_f1_6(class_metrics : dict[str, dict]) -> float :
   F1 : float = 0
   for key,value in class_metrics.items() :
     if key == "other" : 
       continue
     F1 += value["f1"]
   return F1 / len(TARGET_LABELS)

def macro_f1_7(class_metrics : dict[str, dict]) -> float :
   F1 : float = 0
   for value in class_metrics.values() : 
     F1 += value["f1"]
   return F1 / len(LABELS)
     
def confusion_matrix(records: dict[str, dict], gold: dict[str, str]) -> list[list[int]]:
    """
    7x7 raw counts. rows = gold label , columns = predicted label , both in LABELS order.
    counts[i][j] = number of segments with gold LABELS[i] predicted as LABELS[j]
    must raise ValueError if set(records) != set(gold)
    """
    if set(records) != set(gold) :
      raise ValueError("predictions and gold do not have the same seg_ids")

    counts : list[list[int]] = [[0] * len(LABELS) for _ in LABELS]
    for seg_id , gold_label in gold.items() :
      row = LABELS.index(gold_label)
      col = LABELS.index(records[seg_id]["label"])
      counts[row][col] += 1
    return counts

def row_normalised(counts: list[list[int]]) -> list[list[float]]:
    """
    each row divided by its sum , so the diagonal is the recall of that class.
    a row with no gold segments (sum 0) stays all 0.0
    """
    normalised : list[list[float]] = []
    for row in counts :
      row_sum = sum(row)
      if row_sum == 0 :
        normalised.append([0.0] * len(row))
      else :
        normalised.append([value / row_sum for value in row])
    return normalised
