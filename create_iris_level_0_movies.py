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
    sji_files = sorted([f for f in all_files if "sji" in f])
    fuv_files = sorted([f for f in all_files if "fuv" in f])
    nuv_files = sorted([f for f in all_files if "nuv" in f])
    sji_1330_files, sji_1400_files, sji_2796_files, sji_2832_files = (
        l0mr.split_iris_sji_files(sji_files)
    )
    logger.info(
        f"Found {len(sji_files)} SJI files:\n"
        f"1330 - {len(sji_1330_files)} 1400 - {len(sji_1400_files)} 2796 - {len(sji_2796_files)} 2832 - {len(sji_2832_files)}\n"
        f"FUV {len(fuv_files)} files\n"
        f"NUV {len(nuv_files)} files\n"
        f"for {date}",
    )
    date_path = os.path.join(args.outpath, date[0:10])
    os.makedirs(date_path, exist_ok=True)
    for band, fits_files in [
        ("SJI_1330", sji_1330_files),
        ("SJI_1400", sji_1400_files),
        ("SJI_2796", sji_2796_files),
        ("SJI_2832", sji_2832_files),
        ("FUV", fuv_files),
        ("NUV", nuv_files),
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
            logger.info(fullpath + " exists")
