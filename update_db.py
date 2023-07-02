from dotenv import load_dotenv
import os
import psycopg2

def connect_to_db():
    load_dotenv()

    conn = psycopg2.connect(
        dbname=os.getenv("DBNAME"),
        user=os.getenv("DBUSER"),
        password=os.getenv("DBPASS"),
        host=os.getenv("DBHOST"),
        port=os.getenv("DBPORT")
    )

    cur = conn.cursor()
    return conn, cur

def create_tables(cur):
    """
    Creates the necessary tables if they do not exist
    """

    # Execute a query to get a list of tables in the database
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
    """)

    # Fetch all the rows from the result of the query
    tables = cur.fetchall()

    # If there are no tables in the database, create your new table
    if len(tables) == 0:
        # Create the 'spray_xwoba' table
        create_spray_xwoba_table_query = '''
        CREATE TABLE spray_xwoba (
            "Name" TEXT,
            "key_mlbam" INTEGER,
            "PA" INTEGER,
            "wOBA" REAL,
            "xwOBA" REAL,
            "sxwOBA" REAL,
            "diff" REAL,
            "diff%" REAL,
            "BB%" REAL,
            "K%" REAL,
            "HR" INTEGER,
            "LA" REAL,
            "Barrels" REAL,
            "pulled_barrels" INTEGER,
            "year" INTEGER
        )
        '''

        cur.execute(create_spray_xwoba_table_query)

        # Create the 'bbe' table
    create_bbe_table_query = '''
    CREATE TABLE "bbe" (
        "pitch_type" TEXT,
        "game_date" TEXT,
        "release_speed" REAL,
        "release_pos_x" REAL,
        "release_pos_z" REAL,
        "player_name" TEXT,
        "batter" INTEGER,
        "pitcher" INTEGER,
        "events" TEXT,
        "description" TEXT,
        "spin_dir" TEXT,
        "spin_rate_deprecated" TEXT,
        "break_angle_deprecated" TEXT,
        "break_length_deprecated" TEXT,
        "zone" REAL,
        "des" TEXT,
        "game_type" TEXT,
        "stand" TEXT,
        "p_throws" TEXT,
        "home_team" TEXT,
        "away_team" TEXT,
        "type" TEXT,
        "hit_location" TEXT,
        "bb_type" TEXT,
        "balls" INTEGER,
        "strikes" INTEGER,
        "game_year" INTEGER,
        "pfx_x" REAL,
        "pfx_z" REAL,
        "plate_x" REAL,
        "plate_z" REAL,
        "on_3b" TEXT,
        "on_2b" TEXT,
        "on_1b" TEXT,
        "outs_when_up" INTEGER,
        "inning" INTEGER,
        "inning_topbot" TEXT,
        "hc_x" REAL,
        "hc_y" REAL,
        "tfs_deprecated" TEXT,
        "tfs_zulu_deprecated" TEXT,
        "fielder_2" INTEGER,
        "umpire" TEXT,
        "sv_id" TEXT,
        "vx0" REAL,
        "vy0" REAL,
        "vz0" REAL,
        "ax" REAL,
        "ay" REAL,
        "az" REAL,
        "sz_top" REAL,
        "sz_bot" REAL,
        "hit_distance_sc" REAL,
        "launch_speed" REAL,
        "launch_angle" REAL,
        "effective_speed" REAL,
        "release_spin_rate" REAL,
        "release_extension" REAL,
        "game_pk" INTEGER,
        "pitcher.1" INTEGER,
        "fielder_2.1" INTEGER,
        "fielder_3" INTEGER,
        "fielder_4" INTEGER,
        "fielder_5" INTEGER,
        "fielder_6" INTEGER,
        "fielder_7" INTEGER,
        "fielder_8" INTEGER,
        "fielder_9" INTEGER,
        "release_pos_y" REAL,
        "estimated_ba_using_speedangle" REAL,
        "estimated_woba_using_speedangle" REAL,
        "woba_value" REAL,
        "woba_denom" REAL,
        "babip_value" REAL,
        "iso_value" REAL,
        "launch_speed_angle" REAL,
        "at_bat_number" INTEGER,
        "pitch_number" INTEGER,
        "pitch_name" TEXT,
        "home_score" INTEGER,
        "away_score" INTEGER,
        "bat_score" INTEGER,
        "fld_score" INTEGER,
        "post_away_score" INTEGER,
        "post_home_score" INTEGER,
        "post_bat_score" INTEGER,
        "post_fld_score" INTEGER,
        "if_fielding_alignment" FLOAT,
        "of_fielding_alignment" FLOAT,
        "spin_axis" REAL,
        "delta_home_win_exp" REAL,
        "delta_run_exp" REAL,
        "sprint_speed" REAL,
        "target" FLOAT,
        "batter_team" TEXT,
        "pitcher_team" TEXT,
        "stand_L" INTEGER,
        "stand_R" INTEGER,
        "batter_name" TEXT,
        "hc_x_adj" REAL,
        "hc_y_adj" REAL,
        "field_x" REAL,
        "field_y" REAL,
        "theta" REAL,
        "theta_deg" REAL,
        "hit_direction" TEXT,
        "pull" INTEGER,
        "oppo" INTEGER,
        "pulled_barrel" INTEGER,
        "rf_xwoba" REAL,
        "sxwOBA" REAL,
        "sxwoba_probs" TEXT,
        "sxwoba_prob_model" REAL
        )
        '''

    cur.execute(create_bbe_table_query)

def insert_data_into_tables(conn, cur, spray_xwoba_paths, bbe_paths):
    # Insert data into the 'spray_xwoba' table
    for csv_file_path in spray_xwoba_paths:
        with open(csv_file_path, 'r') as f:
            next(f) # Skip the header row
            cur.copy_expert("COPY spray_xwoba FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')", f)
    conn.commit()

    # Insert data into the 'bbe' table
    for csv_file_path in bbe_paths:
        with open(csv_file_path, 'r') as f:
            next(f) # Skip the header row
            cur.copy_expert("COPY bbe FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')", f)
    conn.commit()

def close_connection(cur, conn):
    cur.close()
    conn.close()