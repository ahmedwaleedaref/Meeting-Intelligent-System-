import json

from smi.labels import LABELS

REQUIRED_KEYS = {"seg_id", "label", "scores"}
SCORE_SUM_TOLERANCE = 1e-3


class PredictionFileError(ValueError):
    """Raised when a prediction file breaks the B1 schema."""


def validate_one_parsed_json(record, line_no=None) -> None:
    """
    this function should check :
    1. only 3 keys exist with name seg_id , label , scores
    2. type of value of each one of them  seg_id:str , label:str , scores : dict[str,float] or None ; seg_id and label cannot be empty
    3. the label belong to one of 7 labels
    4. if scores is not None : exactly the 7 label keys , each value a number in [0, 1] , sum is 1 within tolerance
    no need to check id duplication here
    raises PredictionFileError on the first problem found
    """
    where = f"line {line_no}: " if line_no is not None else ""

    # 1. exactly the 3 keys
    if not isinstance(record, dict):
        raise PredictionFileError(f"{where}expected a JSON object, got {type(record).__name__}")
    keys = set(record)
    if keys != REQUIRED_KEYS:
        missing = sorted(REQUIRED_KEYS - keys)
        extra = sorted(keys - REQUIRED_KEYS)
        raise PredictionFileError(f"{where}wrong keys, missing={missing} extra={extra}")

    # 2. types
    seg_id, label, scores = record["seg_id"], record["label"], record["scores"]
    if not isinstance(seg_id, str) or not seg_id:
        raise PredictionFileError(f"{where}seg_id must be a non-empty string, got {seg_id!r}")
    if not isinstance(label, str) or not label:
        raise PredictionFileError(f"{where}label must be a non-empty string, got {label!r}")
    if scores is not None and not isinstance(scores, dict):
        raise PredictionFileError(f"{where}scores must be an object or null, got {type(scores).__name__}")

    # 3. label in the label set
    if label not in LABELS:
        raise PredictionFileError(f"{where}label {label!r} not in {LABELS}")

    # 4. scores
    if scores is None:
        return
    if set(scores) != set(LABELS):
        missing = sorted(set(LABELS) - set(scores))
        extra = sorted(set(scores) - set(LABELS))
        raise PredictionFileError(f"{where}scores has wrong keys, missing={missing} extra={extra}")
    for name, value in scores.items():
        # bool is a subclass of int, so reject it explicitly; NaN fails the range check
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise PredictionFileError(f"{where}score for {name!r} must be a number in [0, 1], got {value!r}")
    total = sum(scores.values())
    if abs(total - 1.0) > SCORE_SUM_TOLERANCE:
        raise PredictionFileError(f"{where}scores sum to {total}, expected 1 ± {SCORE_SUM_TOLERANCE}")

def load_predictions(path) -> dict[str , dict] : 
    records : dict[str , dict] = dict()
    has_scores = None  # set by the first record; a file is all-scores or all-null
    
    with open(path , 'r' , encoding='utf-8') as prediction_file :
        #for each line aka jason in file 
        for line_no ,line in  enumerate(prediction_file, start=1) :
            line = line.strip()
            if not line:
               continue 
            try :
               parsed_record = json.loads(line)
            except json.JSONDecodeError as e:
                raise PredictionFileError(f"line {line_no}: invalid JSON ({e.msg})") from e   
            #from converting it we should do the validations 
            validate_one_parsed_json(parsed_record, line_no)
            seg_id = parsed_record['seg_id']
            if seg_id in records : 
                raise PredictionFileError(f"line {line_no}: duplicate seg_id {seg_id!r}")
            record_has_scores = parsed_record["scores"] is not None
            if has_scores is None:
                has_scores = record_has_scores
            elif record_has_scores != has_scores:
                raise PredictionFileError(f"line {line_no}: mixed file, some records have scores and some are null")
            temp_dict : dict = {
                "label" : parsed_record["label"],
                "scores":parsed_record["scores"]
            }
            records[seg_id] = temp_dict 
        
        return records 