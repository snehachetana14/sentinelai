"""
config.py  -  One place for all settings, file paths and dataset facts.
Every other script imports from here, so if you want to change something
(e.g. number of epochs), change it HERE only.
"""
from pathlib import Path

# ---------------------------------------------------------------- folders
ROOT = Path(__file__).resolve().parent.parent      # the SentinelAI/ folder
DATA_DIR = ROOT / "data"                           # raw + cleaned data
MODEL_DIR = ROOT / "models"                        # saved trained models
RESULT_DIR = ROOT / "results"                      # charts + metric tables
for d in (DATA_DIR, MODEL_DIR, RESULT_DIR):
    d.mkdir(exist_ok=True)

TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"

# ---------------------------------------------------------------- dataset columns
# NSL-KDD files have no header row, so we name the 41 features ourselves,
# plus 'label' (normal / attack name) and 'difficulty' (not used).
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]          # text columns
# 'num_outbound_cmds' is always 0 in NSL-KDD, so it carries no information.
FEATURES = [c for c in COLUMNS if c not in ("label", "difficulty", "num_outbound_cmds")]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

# ---------------------------------------------------------------- attack groups
# ~40 attack names -> 4 attack families (+ Normal).
ATTACK_MAP = {
    "normal": "Normal",
    # DoS: flood a server so real users can't use it
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "mailbomb": "DoS", "processtable": "DoS",
    "udpstorm": "DoS", "worm": "DoS",
    # Probe: scanning the network to find weak spots
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L (Remote-to-Local): outsider trying to get a login on a machine
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "httptunnel": "R2L",
    # U2R (User-to-Root): normal user trying to become admin/root
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}
CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]   # index 0..4
UNKNOWN_LABEL = "Unknown"                           # what the anomaly path reports

# Fixed colour per class (same colour everywhere: report charts + dashboard)
CLASS_COLORS = {
    "Normal": "#2a78d6", "DoS": "#eb6834", "Probe": "#1baf7a",
    "R2L": "#eda100", "U2R": "#e87ba4", UNKNOWN_LABEL: "#4a3aa7",
}

# ---------------------------------------------------------------- training settings
RANDOM_STATE = 42
VAL_SIZE = 0.2                    # 20% of training file kept aside for tuning
SMOTE_TARGETS = {"R2L": 5000, "U2R": 2000}   # grow rare classes to this many rows
N_SELECTED_FEATURES = 60          # features kept for the classifiers (mutual information)

# Which classifier the hybrid engine uses: "auto" = best validation macro-F1,
# or force one of "random_forest", "xgboost", "dnn"
CLASSIFIER_CHOICE = "auto"

DNN_EPOCHS = 15
AE_EPOCHS = 30
BATCH_SIZE = 512

# ---------------------------------------------------------------- hybrid engine settings
CLF_ATTACK_THRESHOLD = 0.5   # classifier fires if P(attack) = 1 - P(Normal) >= this
HYBRID_MODE = "or"           # "or" = alert if either model fires | "weighted" = blended score
HYBRID_WEIGHT = 0.3          # weight of classifier in "weighted" mode (autoencoder gets 1 - w)
HYBRID_CUTOFF = 0.4          # "weighted" mode: alert if blended score >= this
# Why 0.3 / 0.4? The anomaly detector is the one that generalises to unseen attacks, so it
# gets the bigger say. evaluate_hybrid.py prints a sweep over weights & cutoffs so you can
# show the trade-off in your report (results/hybrid_weight_sweep.csv).

API_PORT = 8000
