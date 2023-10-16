import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./consumption_temp.csv")
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# grouped = df.groupby('location')

def plot_overview(df, city):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    df = df[df['location'] == city]
    year_data = df.resample('D').mean()
    # Subplot 1: Plot all data with vertical lines separating the years
    axs[0].plot(year_data.index, year_data['consumption'], label = "dayly mean")
    axs[0].set_title('Full Consumption Data')
    axs[0].grid(True)
    axs[0].set_ylabel('Consumption')
    axs[0].set_xlabel('Week of the Year')
    axs[0].xaxis.set_ticks(year_data.index[::7])
    axs[0].xaxis.set_ticklabels(year_data.index[::7].isocalendar().week, rotation = 50)

    # Add vertical lines to separate the years
    axs[0].axvline(pd.Timestamp(f"{year_data.index.year.unique()[-1]}-01-01"), color='green', linestyle='--')

    axs[0].legend()

    # Subplot 2: Plot the last full month with weeks separated by vertical lines
    month_data = df
    # Find the unique months in the DataFrame
    unique_months = month_data.index.to_period('M').unique()
    # Select the second-to-last month
    last_full_month = unique_months[-2]
    # Filter the DataFrame to get data for the second-to-last month
    last_full_month_data = month_data[month_data.index.to_period('M') == last_full_month]
    month_mean = last_full_month_data.resample('D').mean()

    axs[1].plot(last_full_month_data.index, last_full_month_data['consumption'])
    axs[1].plot(month_mean.index, month_mean['consumption'], label="dayly mean")
    axs[1].set_title(f'Consumption over {last_full_month_data.index[0].month_name()} {last_full_month_data.index[0].year}')
    axs[1].grid(True)
    axs[1].set_ylabel('Consumption')
    axs[1].set_xlabel('Day of the Month')
    axs[1].xaxis.set_ticks(last_full_month_data.index[::24])
    axs[1].xaxis.set_ticklabels(last_full_month_data.index[::24].day)

    # Add vertical lines to separate the weeks
    for week_start in pd.date_range(start=last_full_month_data.index.min(), end=last_full_month_data.index.max(), freq='W-MON'):
        axs[1].axvline(week_start, color='green', linestyle='--')

    axs[1].legend()
    
    # Subplot 3: Plot the time series over a week with days separated by vertical lines
    unique_weeks = df.index.to_period('W').unique()
    # Select the second-to-last month
    last_full_week = unique_weeks[-2]
    # Filter the DataFrame to get data for the second-to-last month
    last_full_week_data = df[df.index.to_period('W') == last_full_week]
    axs[2].plot(last_full_week_data.index, last_full_week_data['consumption'])
    axs[2].set_title(f'Consumption over a Week in {last_full_week_data.index[0].month_name()} {last_full_week_data.index[0].year}')
    axs[2].grid(which='both', linestyle='--', linewidth=0.5)
    axs[2].set_ylabel('Consumption')
    axs[2].set_xlabel('Day of the Month')
    axs[2].minorticks_on()
    axs[2].xaxis.set_ticks(last_full_week_data.index, minor=True)
    axs[2].xaxis.set_ticks(last_full_week_data.index[::24])
    axs[2].xaxis.set_ticklabels(last_full_week_data.index[::24].day)

    # Add vertical lines to separate the days
    for day in pd.date_range(start=last_full_week_data.index.min(), end=last_full_week_data.index.max(), freq='D'):
        axs[2].axvline(pd.Timestamp(day), color='green', linestyle='--')

    # Adjust the layout
    plt.suptitle(f"Consumption of {city}")
    plt.tight_layout()
    plt.savefig(f"overview_{city}.png", dpi=200)
    plt.close()

def plot_consumption_and_temperature(df,location):
    fig, axes = plt.subplots(nrows=2, figsize=(10, 6), sharex=True)

    ax1 = axes[1]
    ax2 = axes[0]  # Create a secondary y-axis

    # Filter data for the current location
    df = df[df['location'] == location]

    months = df.index.to_period('M').unique()
    # Select the second-to-last month
    full_month = months[-2]
    # Filter the DataFrame to get data for the second-to-last month
    data = df[df.index.to_period('M') == full_month]
    mean_data = data.resample('D').mean()

    ax1.plot(mean_data.index, mean_data['consumption'], color = "green")
    ax2.plot(mean_data.index, mean_data['temperature'], color = "blue")
    ax1.grid(axis='y')
    ax2.grid(axis='y')
    ax2.xaxis.set_visible(False)
    ax1.set_ylabel('Consumption')
    ax2.set_ylabel('Temperature')
    ax1.set_xlabel('Day of the Month')
    ax1.xaxis.set_ticks(mean_data.index)
    ax1.xaxis.set_ticklabels(mean_data.index.day)

    # Adjust the layout
    plt.suptitle(f'Consumption vs Temperature of {mean_data.index[0].month_name()} {mean_data.index[0].year} in {location}')
    plt.tight_layout()
    plt.savefig(f"temp_cons_{location}.png", dpi=200)
    plt.close()

# plot_overview(df, 'oslo')
plot_consumption_and_temperature(df, 'oslo')
