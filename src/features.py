import pandas as pd
from src.consts import FEATURE_COLUMNS, ZONE_COLUMNS, TARGET_COLUMN, CATEGORICAL_COLUMNS

def feature_engineering(df) -> pd.DataFrame:
    features = df.copy()
    action_type = df['ACTION_TYPE']

    features['TOTAL_SECONDS_REMAINING'] = (
    (4 - features['PERIOD'].clip(upper=4)) * 720 +
    features['MINUTES_REMAINING'] * 60 +
    features['SECONDS_REMAINING']
    )
    features['3PT_JUMP_SHOT'] = (action_type == '3PT Jump Shot').astype(int)
    features['3PT_PULLUP_JUMP_SHOT'] = (action_type == '3PT Pull-up Jump Shot').astype(int)
    features['3PT_STEP_BACK_JUMP_SHOT'] = (action_type == '3PT Step Back Jump Shot').astype(int)
    features['ALLEY_OOP_DUNK'] = (action_type == 'Alley Oop Dunk').astype(int)
    features['CUTTING_DUNK_SHOT'] = (action_type == 'Cutting Dunk Shot').astype(int)
    features['CUTTING_FINGER_ROLL_LAYUP_SHOT'] = (action_type == 'Cutting Finger Roll Layup Shot').astype(int)
    features['CUTTING_LAYUP_SHOT'] = (action_type == 'Cutting Layup Shot').astype(int)
    features['DRIVING_FINGER_ROLL_LAYUP'] = (action_type == 'Driving Finger Roll Layup').astype(int)
    features['DRIVING_LAYUP'] = (action_type == 'Driving Layup').astype(int)
    features['DRIVING_REVERSE_LAYUP'] = (action_type == 'Driving Reverse Layup').astype(int)
    features['DUNK'] = (action_type == 'Dunk').astype(int)
    features['FINGER_ROLL_LAYUP'] = (action_type == 'Finger Roll Layup').astype(int)
    features['FLOATING_JUMP_SHOT'] = (action_type == 'Floating Jump Shot').astype(int)
    features['HOOK_SHOT'] = (action_type == 'Hook Shot').astype(int)
    features['JUMP_BANK_SHOT'] = (action_type == 'Jump Bank Shot').astype(int)
    features['JUMP_SHOT'] = (action_type == 'Jump Shot').astype(int)
    features['LAYUP'] = (action_type == 'Layup').astype(int)
    features['PULLUP_JUMP_SHOT'] = (action_type == 'Pull-up Jump Shot').astype(int)
    features['PUTBACK_LAYUP'] = (action_type == 'Putback Layup').astype(int)
    features['REVERSE_LAYUP'] = (action_type == 'Reverse Layup').astype(int)
    features['RUNNING_DUNK'] = (action_type == 'Running Dunk').astype(int)
    features['RUNNING_FINGER_ROLL_LAYUP'] = (action_type == 'Running Finger Roll Layup').astype(int)
    features['RUNNING_LAYUP'] = (action_type == 'Running Layup').astype(int)
    features['STEP_BACK_JUMP_SHOT'] = (action_type == 'Step Back Jump Shot').astype(int)
    features['TIP_LAYUP_SHOT'] = (action_type == 'Tip Layup Shot').astype(int)
    features['TURNAROUND_FADEAWAY'] = (action_type == 'Turnaround Fadeaway').astype(int)
    features['TURNAROUND_HOOK_SHOT'] = (action_type == 'Turnaround Hook Shot').astype(int)
    features['TURNAROUND_JUMP_SHOT'] = (action_type == 'Turnaround Jump Shot').astype(int)

    features = pd.get_dummies(features, columns=CATEGORICAL_COLUMNS+ZONE_COLUMNS, dtype=int)
    return features

def get_feature_cols(df) -> list:
    zones_encoded = [col for col in df.columns if col.startswith(tuple(f'{z}_' for z in ZONE_COLUMNS))]
    encoded = [col for col in df.columns if col.startswith(('ACTION_TYPE_', 'SHOT_TYPE_'))]
    base_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
    return base_cols + zones_encoded + encoded

def get_coords(df) -> tuple:
    engineered_df = feature_engineering(df)
    feature_cols = get_feature_cols(engineered_df)
    X = engineered_df[feature_cols]
    y = df[TARGET_COLUMN]
    return X, y