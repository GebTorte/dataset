# /*
# MIT License
#
# Copyright (c) [2022] [CloudSEN12 team]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# R packages --------------------------------------------------------------
library(reticulate)
library(lubridate)
library(dplyr)
library(readr)
library(rgee)
library(stars)
library(sf)
source("utils.R")

# Path with the cloudSEN12 dataset
CLOUDSEN12_DIR <- "/dss/dsshome1/00/di54xat/cloudsen12"

VERSION <- "CLOUDFREE_DEV/"

CLOUDSEN12_PATH <- file.path(CLOUDSEN12_DIR, VERSION)

# 1. Initialize Earth Engine ----------------------------------------------
ee_check()

ee_Initialize(project="earthobserver-cc-476010")
ee_Authenticate()

# 2. Load cloudsen12 initial dataset (after image tile selection) ---------
# SISA: cloudfree dev (head10)
cloudsen12_init <- read.csv("data/cloudsen12_initial_cloudfree_dev.csv") %>% 
  as_tibble()

# 3. Select an CloudSEN12 image patch (IP) --------------------------------
index <- 1
cloudsen12_ip <- cloudsen12_init[index,]

# TODO: loop over all indices?

# 4. Create a cloudSEN12 IP. ----------------------------------------------
cloudsen12_ip_stars <- ip_creator(dataset = cloudsen12_ip, output = CLOUDSEN12_PATH)

# 5. Create metadata object for each IP. ----------------------------------
cloudsen12_ip_metadata <- metadata_creator(
  dataset = cloudsen12_ip,
  raster_ref = cloudsen12_ip_stars,
  output = CLOUDSEN12_PATH
)
