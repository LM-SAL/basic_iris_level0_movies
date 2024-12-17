#!/janus/sanhome/data_ops/miniforge3/envs/sunpy/bin/python

import argparse
import datetime as dt
import glob
import logging
import os

import iris_level_0_movies_routines as l0mr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Creating IRIS Level 0 Movies")
parser.add_argument(
    "--begin",
    "-b",
    help="Start yyyy/mm/dd/Hhhmm; ignored if --hours is set.",
    default="2024/12/06/H0000",
)
parser.add_argument(
    "--end",
    "-e",
    help="End yyyy/mm/dd/Hhhmm; ignored if --hours is set.",
    default="2024/12/06/H0100",
)
parser.add_argument(
    "--hours",
    help="Try to create movies that many hours into the past from now",
    type=int,
)
parser.add_argument(
    "--l0path",
    help="Path to Level 0 data",
    default="/irisa/data/level0/",
)
parser.add_argument("--outpath", "-o", help="Output path", default="./")
args = parser.parse_args()

if args.hours is None:
    dates = l0mr.get_date_range(args.begin, args.end)
else:
    tnow = dt.datetime.now(dt.timezone.utc)
    str_end = tnow.strftime("%Y/%m/%d/H%H00")
    str_begin = (tnow - dt.timedelta(hours=args.hours)).strftime("%Y/%m/%d/H%H00")
    logger.info(f"Trying to make movies from {str_begin} to {str_end}")
    dates = l0mr.get_date_range(str_begin, str_end)

for date in dates:
    all_files = glob.glob(args.l0path + date + "/*.fits")
    sji_files = [f for f in all_files if "sji" in f]
    fuv_files = [f for f in all_files if "fuv" in f]
    nuv_files = [f for f in all_files if "nuv" in f]
    logger.info(
        f"Found {len(sji_files)} SJI files, {len(fuv_files)} FUV files, and {len(nuv_files)} NUV files for {date}",
    )
    date_path = os.path.join(args.outpath, date[0:10])
    os.makedirs(date_path, exist_ok=True)
    for band, fits_files in [
        ("sji", sji_files),
        ("fuv", fuv_files),
        ("nuv", nuv_files),
    ]:
        if len(fits_files) == 0:
            logger.info(f"No {band} files found for {date}")
            continue
        fullpath = os.path.join(date_path, f"{date[11:16]}_{band}.mp4")
        if not os.path.exists(fullpath):
            movie_maker = l0mr.FITSMovieMaker(
                fits_files=fits_files,
                output_path=fullpath,
            )
            movie_maker.create_movie()
        else:
            logger.info(fullpath + " exist.")
