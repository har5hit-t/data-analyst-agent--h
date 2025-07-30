import duckdb
import matplotlib.pyplot as plt
import base64
import io
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/api/")
async def analyze(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        content = (await file.read()).decode("utf-8")

        # Extract URL
        lines = content.strip().splitlines()
        url_line = next((line for line in lines if "http" in line), None)
        if not url_line:
            return JSONResponse(content={"error": "URL not found in file"}, status_code=400)

        url = url_line.strip()

        # Scrape HTML and parse tables
        res = requests.get(url)
        tables = pd.read_html(res.text)

        # Find correct table
        df = None
        for table in tables:
            if "Title" in table.columns and "Worldwide gross" in table.columns:
                df = table.copy()
                break

        if df is None:
            return JSONResponse(content={"error": "Required table not found"}, status_code=400)

        # Clean data
        df["Worldwide gross"] = df["Worldwide gross"].astype(str).str.replace(r"[^\d.]", "", regex=True).astype(float)
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
        df["Peak"] = pd.to_numeric(df["Peak"], errors="coerce")

        # Answer 1: $2bn movies before 2020
        q1 = df[(df["Worldwide gross"] >= 2_000_000_000) & (df["Year"] < 2020)].shape[0]

        # Answer 2: Earliest $1.5bn+ film
        q2_df = df[df["Worldwide gross"] > 1_500_000_000]
        q2 = q2_df[q2_df["Year"] == q2_df["Year"].min()]["Title"].values[0]

        # Answer 3: Correlation between Rank and Peak
        corr_df = df[["Rank", "Peak"]].dropna()
        q3 = round(corr_df["Rank"].corr(corr_df["Peak"]), 3)

        # Answer 4: Plot with valid Rank/Peak only
        clean_df = df[["Rank", "Peak"]].dropna()
        fig, ax = plt.subplots()
        ax.scatter(clean_df["Rank"], clean_df["Peak"], color="blue")

        m, b = np.polyfit(clean_df["Rank"], clean_df["Peak"], 1)
        ax.plot(clean_df["Rank"], m * clean_df["Rank"] + b, linestyle='dotted', color='red')
        ax.set_xlabel("Rank")
        ax.set_ylabel("Peak")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{img_str}"

        if len(data_uri) > 100_000:
            data_uri = "data:image/png;base64," + img_str[:100000 - 22]

        return JSONResponse(content=[
            str(q1),
            q2,
            str(q3),
            data_uri
        ])

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
