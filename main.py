from imports import *
from config import *
from data_retrieval import get_season_data, get_fg_stats
from preprocess_data import preprocess_data, create_target_variable, get_sprint_speed
from train_models import train_model
from results_processing import calculate_expected_xwoba, postprocess_data
from database_operations import connect_to_db, create_tables, insert_data_into_tables, close_connection

def main():
    years_to_query = [2021, 2022, 2023]
    start_end = pd.read_csv("start_end_dates.csv")
    print(start_end.info())

    for year in years_to_query:
        season_df = get_season_data(year)
        season_df = get_sprint_speed(season_df, year)
        season_df = create_target_variable(season_df)
        df, bbe = preprocess_data(season_df)
        bbe = train_model(year, bbe)
        leaders, sxwoba_leaders = calculate_expected_xwoba(df, bbe, year)
        df, leaders, bbe = postprocess_data(df, leaders, bbe, year)

        # Save data to csv
        bbe.to_csv(f"statcast_data/bbe/bbe_{year}.csv", index=False)
        leaders.to_csv(f"leaders/spray_xwoba_{year}.csv", index=False)
        if year == 2023:
            df.to_csv("statcast_data/current.csv")
    # Connect to the database
    conn, cur = connect_to_db()

    # Create the tables
    create_tables(cur)

    # Define the paths to your CSV files
    spray_xwoba_dir = 'C:/Users/wampl/sxwOBA/leaders/'
    bbe_dir = 'C:/Users/wampl/sxwOBA/statcast_data/bbe/'

    csv_file_paths_spray_xwoba = glob.glob(spray_xwoba_dir + '*.csv')
    csv_file_paths_bbe = glob.glob(bbe_dir + '*.csv')

    # Insert the data into the tables
    insert_data_into_tables(
        conn,
        cur,
        csv_file_paths_spray_xwoba,
        csv_file_paths_bbe
    )

    # Close the connection to the database
    close_connection(cur, conn)


if __name__ == "__main__":
    main()