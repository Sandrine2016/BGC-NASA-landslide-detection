from data.downloader import reddit
import data.articles as articles


def get_articles(start_date, end_date):
    df = reddit.download_posts(start_date, end_date)
    df = articles.add_articles_to_df(df)
    df = articles.filter_invalid_articles(df)
    df = articles.filter_negative_articles(df)

    df = df.dropna(
        subset=["article_publish_date"]
    )

    return df
