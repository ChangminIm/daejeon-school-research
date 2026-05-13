"""송부자료(xlsx)에서 초·중학교 데이터 추출

좌측 25열: 초등학교, 우측: 중학교 구조 처리
"""
import pandas as pd
from src.config import DATA_RAW, DATA_PROCESSED


def is_school_row(row, school_col):
    name = row[school_col]
    if pd.isna(name) or not isinstance(name, str):
        return False
    if "계" in name and "교" in name and ")" in name:
        return False
    if name.startswith(("동부", "서부", "본교")) or name == "합계":
        return False
    return True


def parse(input_path=None):
    if input_path is None:
        input_path = DATA_RAW / "송부_자료_1차_.xlsx"

    df = pd.read_excel(input_path, sheet_name="학교 주소 및 학생수", header=None)

    # 초등학교 추출 (좌측)
    elem = []
    for i in range(8, len(df)):
        row = df.iloc[i].tolist()
        if is_school_row(row, 3):
            elem.append({
                "학교급": "초", "설립": row[1], "구": row[2],
                "학교명": row[3], "주소": row[4], "구분": row[5],
                "학생수합계": row[22],
            })

    # 중학교 추출 (우측)
    mid = []
    for i in range(8, len(df)):
        row = df.iloc[i].tolist()
        if pd.notna(row[28]) and is_school_row(row, 28):
            mid.append({
                "학교급": "중", "설립": row[26], "구": row[27],
                "학교명": row[28], "주소": row[29], "구분": row[30],
                "학생수합계": row[41],
            })

    result = pd.concat([pd.DataFrame(elem), pd.DataFrame(mid)], ignore_index=True)
    result["학생수합계"] = pd.to_numeric(result["학생수합계"], errors="coerce").fillna(0).astype(int)
    return result


if __name__ == "__main__":
    df = parse()
    out = DATA_PROCESSED / "schools.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"✅ {len(df)}개 학교 → {out}")
    print(df["구"].value_counts().to_string())
