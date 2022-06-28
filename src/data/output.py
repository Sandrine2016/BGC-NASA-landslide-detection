def get_formatted_df(df, predictions):

    df["landslide_category"] = predictions[3]
    df["landslide_trigger"] = predictions[4]
    df["location_description"] = [p.name for p in predictions[0]]
    df["latitude"] = [p.lat for p in predictions[0]]
    df["longitude"] = [p.lng for p in predictions[0]]
    df["location_accuracy"] = [p.radius for p in predictions[0]]
    df["interval_start"] = [p.interval_start for p in predictions[1]]
    df["interval_end"] = [p.interval_end for p in predictions[1]]
    df["discrete_date"] = [p.discrete_date for p in predictions[1]]
    df["event_date_accuracy"] = [p.confidence for p in predictions[1]]
    df["fatality_count"] = predictions[2]


    df = df.dropna(
        subset=["location", "latitude", "longitude", "interval_start", "interval_end"]
    )

    df = df.rename(
        columns={
            "created_utc": "reddit_created_utc",
            "selftext": "reddit_text",
            "headline": "reddit_title",
            "article_title": "event_description",
        }
    )

    df = df.drop(
        columns=[
            "created",
            "score",
            "keyword",
            "text",
            "article_summary",
            "sub_text",
            "lang",
            "similarity",
            "discrete_date",
        ]
    )

    return df

    # nasa_df = pd.read_csv(
    #     os.path.join(DATA_PATH, "nasa", "nasa_global_landslide_catalog_point.csv")
    # )
    # final_df = format_column_nasa_format(
    #     remove_duplicates(filtered_reddit_articles_df, nasa_df)
    # )
    # final_df.to_csv(os.path.join(DATA_PATH, "output", "results_bert.csv"))
