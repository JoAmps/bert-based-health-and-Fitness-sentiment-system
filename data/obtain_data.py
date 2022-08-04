import pandas as pd
from tqdm import tqdm
import seaborn as sns
from google_play_scraper import Sort, reviews, app


sns.set(style='whitegrid', palette='muted', font_scale=1.2)

app_packages = [
    'steptracker.healthandfitness.walkingtracker.pedometer',
    'homeworkout.homeworkouts.noequipment',
    'dance.fit.zumba.weightloss.danceburn',
    'women.workout.female.fitness',
    'com.droidinfinity.healthplus',
    'com.fiton.android',
    'pedometer.steptracker.calorieburner.stepcounter',
    'com.myfitnesspal.android',
    'com.fitbit.FitbitMobile',
    'com.runtastic.android.results.lite',
    'menloseweight.loseweightappformen.weightlossformen',
    'loseweightapp.loseweightappforwomen.womenworkoutathome',
    'sixpack.sixpackabs.absworkout',
    'losebellyfat.flatstomach.absworkout.fatburning',
    'com.popularapp.thirtydayfitnesschallenge',
    'net.workout.lose.weight.fitness.fit.coach',
    'com.kaylaitsines.sweatwithkayla',
    'com.nike.ntc',
    'buttocksworkout.hipsworkouts.forwomen.legworkout',
    'loseweight.weightloss.workout.fitness',
    'com.freeletics.lite',
    'com.fitifyworkouts.bodyweight.workoutapp',
    'com.dailyyoga.inc',
    'com.mapmyfitness.android2',
    'fitness.online.app',
    'com.runtastic.android',
    'com.workout.workout',
    'fat.burnning.plank.fitness.loseweight',
    'digifit.virtuagym.client.android',
    'homeworkout.homeworkouts.workoutathome.musclebuilding'
]


def obtain_app_details():
    app_details = []

    for detail in tqdm(app_packages):
        details = app(detail, lang='en', country='us')
        del details['comments']
        app_details.append(details)
        return app_details


def format_title(title):
    title_index = title.find(':') if title.find(':') != -1 else title.find('-')
    if title_index != -1:
        title = title[:title_index]
    return title[:8]


def extract_ratings():
    app_reviews = []

    for ap in tqdm(app_packages):
        for score in list(range(1, 6)):
            for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
                rvs, _ = reviews(
                    ap,
                    lang='en',
                    country='us',
                    sort=sort_order,
                    count=200 if score == 3 else 100,
                    filter_score_with=score
                )
                for r in rvs:
                    r['sortOrder'] = 'most_relevant' if sort_order\
                         == Sort.MOST_RELEVANT else 'newest'
                    r['appId'] = ap

                    app_reviews.extend(rvs)
                    return app_reviews


def store_data(app_reviews):
    app_reviews_df = pd.DataFrame(app_reviews)
    app_reviews_df = app_reviews_df.drop_duplicates()
    app_reviews_df.to_csv('app_reviews.csv', index=None, header=True)


if __name__ == '__main__':
    obtain_app_details()
    print('1')
    app_reviews = extract_ratings()
    print('2')
    store_data(app_reviews)
