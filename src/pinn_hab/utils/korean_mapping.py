"""
Korean to English variable name mapping for water quality data.
"""

from typing import List

# ---- KOREAN TO ENGLISH VARIABLE NAME MAPPING ----

KOREAN_TO_ENGLISH_MAP = {
    '수온 (℃)': 'Water Temperature (°C)',
    '수소이온농도': 'pH',
    '전기전도도 (μS/cm)': 'Conductivity (μS/cm)',
    '용존산소 (mg/L)': 'DO (mg/L)',
    '탁도 (NTU)': 'Turbidity (NTU)',
    '총유기탄소 (mg/L)': 'TOC (mg/L)',
    '클로로필-a (mg/㎥)': 'Chl-a (mg/m³)',
    '염화메틸렌 (μg/L)': 'Chloromethane (μg/L)',
    '1.1.1-트리클로로에테인 (μg/L)': '1,1,1-Trichloroethane (μg/L)',
    '사염화탄소 (μg/L)': 'Carbon tetrachloride (μg/L)',
    '트리클로로에틸렌 (μg/L)': 'Trichloroethylene (μg/L)',
    '테트라클로로에틸렌 (μg/L)': 'Tetrachloroethylene (μg/L)',
    '벤젠 (μg/L)': 'Benzene (μg/L)',
    '톨루엔 (μg/L)': 'Toluene (μg/L)',
    '에틸벤젠 (μg/L)': 'Ethylbenzene (μg/L)',
    'm,p-자일렌 (μg/L)': 'm,p-Xylene (μg/L)',
    'o-자일렌 (μg/L)': 'o-Xylene (μg/L)',
    '[ECD]염화메틸렌 (μg/L)': '[ECD]Chloromethane (μg/L)',
    '[ECD]1.1.1-트리클로로에테인 (μg/L)': '[ECD]1,1,1-Trichloroethane (μg/L)',
    '[ECD]사염화탄소 (μg/L)': '[ECD]Carbon tetrachloride (μg/L)',
    '[ECD]트리클로로에틸렌 (μg/L)': '[ECD]Trichloroethylene (μg/L)',
    '[ECD]테트라클로로에틸렌 (μg/L)': '[ECD]Tetrachloroethylene (μg/L)',
    '총질소 (mg/L)': 'TN (mg/L)',
    '총인 (mg/L)': 'TP (mg/L)',
    '측정일': 'Date',
    '측정소': 'Site',
    '수계': 'Basin',
    '측정소명': 'Site Name',
    '시작연도': 'Start Year',
    '종료연도': 'End Year',
    '파일명': 'Filename'
}


def translate_feature_names(feature_names: List[str]) -> List[str]:
    """
    Translate Korean feature names to English for visualization.
    
    Args:
        feature_names: List of feature names (may contain Korean)
        
    Returns:
        List of translated feature names in English
    """
    translated = []
    
    for name in feature_names:
        # Check if exact match exists
        if name in KOREAN_TO_ENGLISH_MAP:
            translated_name = KOREAN_TO_ENGLISH_MAP[name]
        else:
            # Try partial match for engineered features
            translated_name = name
            for kr, en in KOREAN_TO_ENGLISH_MAP.items():
                if kr in name:
                    # Replace Korean part with English
                    translated_name = name.replace(kr, en)
                    break
            if translated_name == name:
                translated_name = name  # Keep original if no match
        
        # Remove [ECD] prefix if present
        if '[ECD]' in translated_name:
            translated_name = translated_name.replace('[ECD]', '').strip()
        
        # Fix spacing issues
        translated_name = translated_name.replace('μ g/', 'μg/').replace('μ g/', 'μg/')
        translated_name = translated_name.replace('° C', '°C')
        
        # Fix "ma7" to "7-day moving avg" if present
        if 'ma7' in translated_name.lower():
            translated_name = translated_name.replace('ma7', '7-day moving avg')
            translated_name = translated_name.replace('(7-day avg)', '(7-day moving avg)')
        
        translated.append(translated_name)
    
    return translated
