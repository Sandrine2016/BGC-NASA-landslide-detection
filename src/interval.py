


def get_config_interval():
    """
    Reads config file to extract appropriate
    date interval to extract articles from.

    Returns
    -------
    tuple
        (start_date, end_date)
    """

    with open(os.path.join(MAIN_PATH, "config.json")) as f:
        config = json.load(f)
    if config["interval"]["default"] == "yes":
        log = []
        with open(os.path.join(MAIN_PATH, "history.log")) as f:
            for line in f:
                log.append(parser.parse(line.strip().split("\t")[2]))
        if len(log) == 0:
            start_date = datetime(2019, 1, 1)
        else:
            log.sort()
            start_date = log[-1]

        end_date = datetime.now()

    else:
        if config["interval"]["start"]["date"]:
            start_date = parser.parse(config["interval"]["start"]["date"])
        else:
            start_date = datetime(2019, 1, 1)

        if config["interval"]["end"]["now"] == "yes":
            end_date = datetime.now()
        elif config["interval"]["end"]["date"]:
            end_date = parser.parse(config["interval"]["end"]["date"])
        else:
            end_date = datetime.now()

    with open(os.path.join(MAIN_PATH, "history.log"), "a+") as f:
        f.write("\t".join([str(datetime.now()), str(start_date), str(end_date)]) + "\n")

    return start_date, end_date
