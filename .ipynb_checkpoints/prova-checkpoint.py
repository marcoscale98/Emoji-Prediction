{
 "cells": [
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "problema sugli embeddings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Including required python libraries used in this project\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import emoji\n",
    "\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, Input, Dropout, SimpleRNN,LSTM, Activation\n",
    "from keras.utils import np_utils\n",
    "\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# reading train emoji and test emoji data which are attached in the repo\n",
    "train = pd.read_json(r'D:\\Marco\\Universita\\3ANNO\\Tesi\\emoji-prediction\\data_input\\evalita_test.json',orient=\"records\", lines=True, nrows=15)\n",
    "test = pd.read_json(r'D:\\Marco\\Universita\\3ANNO\\Tesi\\emoji-prediction\\data_input\\evalita_train.json',orient=\"records\", lines=True,nrows=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "#ci servirà per trasformare le righe del training set\n",
    "def trasform_rows(row):\n",
    "    emoji_label = row['label']\n",
    "    emoji_label = \":\"+emoji_label+\":\"\n",
    "    if emoji_label not in emoji_dict.values():\n",
    "        key = len(emoji_dict.keys())\n",
    "        emoji_dict[key] = emoji_label\n",
    "        return key"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text_no_emoji</th>\n",
       "      <th>uid</th>\n",
       "      <th>ground_truth_label</th>\n",
       "      <th>tid</th>\n",
       "      <th>created_at</th>\n",
       "      <th>tweet_id</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>@eallora6 io desideravo il meglio e ho sposato...</td>\n",
       "      <td>227841404</td>\n",
       "      <td>winking_face</td>\n",
       "      <td>ITAMOJI_test_1</td>\n",
       "      <td>2015-08-18 23:46:16+00:00</td>\n",
       "      <td>633787023699148800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>#mascoloaformentera \\nBen sticazzi ahaha sei i...</td>\n",
       "      <td>3429936850</td>\n",
       "      <td>winking_face</td>\n",
       "      <td>ITAMOJI_test_2</td>\n",
       "      <td>2017-08-03 21:45:16+00:00</td>\n",
       "      <td>893226284242874368</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>@GLPini @sscnapoli Non ha torto. La squadra di...</td>\n",
       "      <td>2294138965</td>\n",
       "      <td>winking_face</td>\n",
       "      <td>ITAMOJI_test_3</td>\n",
       "      <td>2018-01-22 07:51:03+00:00</td>\n",
       "      <td>955347059078500352</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Su' questo hai ragione https://t.co/qfDawgI2Y9</td>\n",
       "      <td>395361626</td>\n",
       "      <td>winking_face</td>\n",
       "      <td>ITAMOJI_test_4</td>\n",
       "      <td>2017-12-19 16:39:07+00:00</td>\n",
       "      <td>943158765574066176</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Visita archeologica...!!! @ Colosseo https://t...</td>\n",
       "      <td>856817140715880448</td>\n",
       "      <td>winking_face</td>\n",
       "      <td>ITAMOJI_test_5</td>\n",
       "      <td>2017-09-14 13:11:01+00:00</td>\n",
       "      <td>908317161541664768</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                       text_no_emoji                 uid  \\\n",
       "0  @eallora6 io desideravo il meglio e ho sposato...           227841404   \n",
       "1  #mascoloaformentera \\nBen sticazzi ahaha sei i...          3429936850   \n",
       "2  @GLPini @sscnapoli Non ha torto. La squadra di...          2294138965   \n",
       "3     Su' questo hai ragione https://t.co/qfDawgI2Y9           395361626   \n",
       "4  Visita archeologica...!!! @ Colosseo https://t...  856817140715880448   \n",
       "\n",
       "  ground_truth_label             tid                created_at  \\\n",
       "0       winking_face  ITAMOJI_test_1 2015-08-18 23:46:16+00:00   \n",
       "1       winking_face  ITAMOJI_test_2 2017-08-03 21:45:16+00:00   \n",
       "2       winking_face  ITAMOJI_test_3 2018-01-22 07:51:03+00:00   \n",
       "3       winking_face  ITAMOJI_test_4 2017-12-19 16:39:07+00:00   \n",
       "4       winking_face  ITAMOJI_test_5 2017-09-14 13:11:01+00:00   \n",
       "\n",
       "             tweet_id  \n",
       "0  633787023699148800  \n",
       "1  893226284242874368  \n",
       "2  955347059078500352  \n",
       "3  943158765574066176  \n",
       "4  908317161541664768  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Checking data by showing first 5 rows of the train data\n",
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>tid</th>\n",
       "      <th>uid</th>\n",
       "      <th>created_at</th>\n",
       "      <th>text_no_emoji</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>622416920701054976</td>\n",
       "      <td>3115912511</td>\n",
       "      <td>2015-07-18 14:45:32+00:00</td>\n",
       "      <td>#Noiaaa#goro#aspettandolasera @ Porto di Goro ...</td>\n",
       "      <td>red_heart</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>737712884965363712</td>\n",
       "      <td>423498157</td>\n",
       "      <td>2016-05-31 18:30:32+00:00</td>\n",
       "      <td>e niente, nonostante i casini e gli impegni di...</td>\n",
       "      <td>red_heart</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>671023999057567744</td>\n",
       "      <td>488427533</td>\n",
       "      <td>2015-11-29 17:52:43+00:00</td>\n",
       "      <td>#Faccebuffe #friends #friendship #saturdaynigh...</td>\n",
       "      <td>red_heart</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>650002923393548288</td>\n",
       "      <td>3041411470</td>\n",
       "      <td>2015-10-02 17:42:28+00:00</td>\n",
       "      <td>Un nuovo post è ora online su http://t.co/h6pK...</td>\n",
       "      <td>red_heart</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>628245096694202368</td>\n",
       "      <td>1693227686</td>\n",
       "      <td>2015-08-03 16:44:37+00:00</td>\n",
       "      <td>@vogliosoloriker video e magari iscriverti?Ho ...</td>\n",
       "      <td>red_heart</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  tid         uid                created_at  \\\n",
       "0  622416920701054976  3115912511 2015-07-18 14:45:32+00:00   \n",
       "1  737712884965363712   423498157 2016-05-31 18:30:32+00:00   \n",
       "2  671023999057567744   488427533 2015-11-29 17:52:43+00:00   \n",
       "3  650002923393548288  3041411470 2015-10-02 17:42:28+00:00   \n",
       "4  628245096694202368  1693227686 2015-08-03 16:44:37+00:00   \n",
       "\n",
       "                                       text_no_emoji      label  \n",
       "0  #Noiaaa#goro#aspettandolasera @ Porto di Goro ...  red_heart  \n",
       "1  e niente, nonostante i casini e gli impegni di...  red_heart  \n",
       "2  #Faccebuffe #friends #friendship #saturdaynigh...  red_heart  \n",
       "3  Un nuovo post è ora online su http://t.co/h6pK...  red_heart  \n",
       "4  @vogliosoloriker video e magari iscriverti?Ho ...  red_heart  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Checking data by showing first 5 rows of the test data\n",
    "test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "#creo un dizionario con tutte le emoji presenti nel training set e aggiungo una colonna 'key_emoji'\n",
    "emoji_dict = {}\n",
    "train.rename({\"ground_truth_label\":\"label\"}, axis=1, inplace=True)\n",
    "train[\"key_emoji\"] = train.apply(trasform_rows,axis=1)\n",
    "test[\"key_emoji\"] = test.apply(trasform_rows,axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text_no_emoji</th>\n",
       "      <th>uid</th>\n",
       "      <th>label</th>\n",
       "      <th>tid</th>\n",
       "      <th>created_at</th>\n",
       "      <th>tweet_id</th>\n",
       "      <th>key_emoji</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>@eallora6 io desideravo il meglio e ho sposato...</td>\n",
       "      <td>227841404</td>\n",
       "      <td>winking_face</td>\n",
       "      <td>ITAMOJI_test_1</td>\n",
       "      <td>2015-08-18 23:46:16+00:00</td>\n",
       "      <td>633787023699148800</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>#mascoloaformentera \\nBen sticazzi ahaha sei i...</td>\n",
       "      <td>3429936850</td>\n",
       "      <td>winking_face</td>\n",
       "      <td>ITAMOJI_test_2</td>\n",
       "      <td>2017-08-03 21:45:16+00:00</td>\n",
       "      <td>893226284242874368</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>@GLPini @sscnapoli Non ha torto. La squadra di...</td>\n",
       "      <td>2294138965</td>\n",
       "      <td>winking_face</td>\n",
       "      <td>ITAMOJI_test_3</td>\n",
       "      <td>2018-01-22 07:51:03+00:00</td>\n",
       "      <td>955347059078500352</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Su' questo hai ragione https://t.co/qfDawgI2Y9</td>\n",
       "      <td>395361626</td>\n",
       "      <td>winking_face</td>\n",
       "      <td>ITAMOJI_test_4</td>\n",
       "      <td>2017-12-19 16:39:07+00:00</td>\n",
       "      <td>943158765574066176</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Visita archeologica...!!! @ Colosseo https://t...</td>\n",
       "      <td>856817140715880448</td>\n",
       "      <td>winking_face</td>\n",
       "      <td>ITAMOJI_test_5</td>\n",
       "      <td>2017-09-14 13:11:01+00:00</td>\n",
       "      <td>908317161541664768</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                       text_no_emoji                 uid  \\\n",
       "0  @eallora6 io desideravo il meglio e ho sposato...           227841404   \n",
       "1  #mascoloaformentera \\nBen sticazzi ahaha sei i...          3429936850   \n",
       "2  @GLPini @sscnapoli Non ha torto. La squadra di...          2294138965   \n",
       "3     Su' questo hai ragione https://t.co/qfDawgI2Y9           395361626   \n",
       "4  Visita archeologica...!!! @ Colosseo https://t...  856817140715880448   \n",
       "\n",
       "          label             tid                created_at            tweet_id  \\\n",
       "0  winking_face  ITAMOJI_test_1 2015-08-18 23:46:16+00:00  633787023699148800   \n",
       "1  winking_face  ITAMOJI_test_2 2017-08-03 21:45:16+00:00  893226284242874368   \n",
       "2  winking_face  ITAMOJI_test_3 2018-01-22 07:51:03+00:00  955347059078500352   \n",
       "3  winking_face  ITAMOJI_test_4 2017-12-19 16:39:07+00:00  943158765574066176   \n",
       "4  winking_face  ITAMOJI_test_5 2017-09-14 13:11:01+00:00  908317161541664768   \n",
       "\n",
       "   key_emoji  \n",
       "0        0.0  \n",
       "1        NaN  \n",
       "2        NaN  \n",
       "3        NaN  \n",
       "4        NaN  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>tid</th>\n",
       "      <th>uid</th>\n",
       "      <th>created_at</th>\n",
       "      <th>text_no_emoji</th>\n",
       "      <th>label</th>\n",
       "      <th>key_emoji</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>622416920701054976</td>\n",
       "      <td>3115912511</td>\n",
       "      <td>2015-07-18 14:45:32+00:00</td>\n",
       "      <td>#Noiaaa#goro#aspettandolasera @ Porto di Goro ...</td>\n",
       "      <td>red_heart</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>737712884965363712</td>\n",
       "      <td>423498157</td>\n",
       "      <td>2016-05-31 18:30:32+00:00</td>\n",
       "      <td>e niente, nonostante i casini e gli impegni di...</td>\n",
       "      <td>red_heart</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>671023999057567744</td>\n",
       "      <td>488427533</td>\n",
       "      <td>2015-11-29 17:52:43+00:00</td>\n",
       "      <td>#Faccebuffe #friends #friendship #saturdaynigh...</td>\n",
       "      <td>red_heart</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>650002923393548288</td>\n",
       "      <td>3041411470</td>\n",
       "      <td>2015-10-02 17:42:28+00:00</td>\n",
       "      <td>Un nuovo post è ora online su http://t.co/h6pK...</td>\n",
       "      <td>red_heart</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>628245096694202368</td>\n",
       "      <td>1693227686</td>\n",
       "      <td>2015-08-03 16:44:37+00:00</td>\n",
       "      <td>@vogliosoloriker video e magari iscriverti?Ho ...</td>\n",
       "      <td>red_heart</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  tid         uid                created_at  \\\n",
       "0  622416920701054976  3115912511 2015-07-18 14:45:32+00:00   \n",
       "1  737712884965363712   423498157 2016-05-31 18:30:32+00:00   \n",
       "2  671023999057567744   488427533 2015-11-29 17:52:43+00:00   \n",
       "3  650002923393548288  3041411470 2015-10-02 17:42:28+00:00   \n",
       "4  628245096694202368  1693227686 2015-08-03 16:44:37+00:00   \n",
       "\n",
       "                                       text_no_emoji      label  key_emoji  \n",
       "0  #Noiaaa#goro#aspettandolasera @ Porto di Goro ...  red_heart        1.0  \n",
       "1  e niente, nonostante i casini e gli impegni di...  red_heart        NaN  \n",
       "2  #Faccebuffe #friends #friendship #saturdaynigh...  red_heart        NaN  \n",
       "3  Un nuovo post è ora online su http://t.co/h6pK...  red_heart        NaN  \n",
       "4  @vogliosoloriker video e magari iscriverti?Ho ...  red_heart        NaN  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0 😉\n",
      "1 ❤\n"
     ]
    }
   ],
   "source": [
    "# Printing each emoji icon by emojizing each emoji\n",
    "for ix in emoji_dict.keys():\n",
    "    print (ix,end=\" \")\n",
    "    print (emoji.emojize(emoji_dict[ix], use_aliases=True))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(15,) (15,) (10,) (10,)\n",
      "-------------------------\n",
      "@eallora6 io desideravo il meglio e ho sposato il peggio 0.0\n"
     ]
    }
   ],
   "source": [
    "# Creating training and testing data\n",
    "X_train = train['text_no_emoji']\n",
    "Y_train = train['key_emoji']\n",
    "\n",
    "X_test = test['text_no_emoji']\n",
    "Y_test = test['key_emoji']\n",
    "\n",
    "print (X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)\n",
    "print (\"-------------------------\")\n",
    "print (X_train[0],Y_train[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-11-7fb8399e9b6d>:3: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  X_train[ix] = X_train[ix].split()\n",
      "<ipython-input-11-7fb8399e9b6d>:7: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  X_test[ix] = X_test[ix].split()\n"
     ]
    }
   ],
   "source": [
    "# Splitting the train data from sentences to words\n",
    "for ix in range(X_train.shape[0]):\n",
    "    X_train[ix] = X_train[ix].split()\n",
    "\n",
    "# Splitting the test data from sentences to words\n",
    "for ix in range(X_test.shape[0]):\n",
    "    X_test[ix] = X_test[ix].split()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "ename": "IndexError",
     "evalue": "index -2147483648 is out of bounds for axis 1 with size 2",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mIndexError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-12-c41e1d518510>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m# Converting labels into categorical form\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0mY_train\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp_utils\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mto_categorical\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mY_train\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mnum_classes\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0memoji_dict\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m~\\.conda\\envs\\emoji-prediction\\lib\\site-packages\\tensorflow\\python\\keras\\utils\\np_utils.py\u001b[0m in \u001b[0;36mto_categorical\u001b[1;34m(y, num_classes, dtype)\u001b[0m\n\u001b[0;32m     76\u001b[0m   \u001b[0mn\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     77\u001b[0m   \u001b[0mcategorical\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mzeros\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mn\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mnum_classes\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 78\u001b[1;33m   \u001b[0mcategorical\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0marange\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mn\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     79\u001b[0m   \u001b[0moutput_shape\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0minput_shape\u001b[0m \u001b[1;33m+\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mnum_classes\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     80\u001b[0m   \u001b[0mcategorical\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mreshape\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcategorical\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0moutput_shape\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mIndexError\u001b[0m: index -2147483648 is out of bounds for axis 1 with size 2"
     ]
    }
   ],
   "source": [
    "# Converting labels into categorical form\n",
    "Y_train = np_utils.to_categorical(Y_train, num_classes=len(emoji_dict))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Now checking the above conversion by printing train and test data at 0th index\n",
    "print (X_train[0],Y_train[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# To check maximum length of sentence in training data\n",
    "np.unique(np.array([len(ix) for ix in X_train]) , return_counts=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# To check maximum length of senetence in testing data\n",
    "np.unique(np.array([len(ix) for ix in X_test]) , return_counts=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Creating  embeddings dictionary with key = word and value = list of words in glove vector\n",
    "embeddings_index = {}\n",
    "\n",
    "f = open(r'D:\\Marco\\Universita\\3ANNO\\Tesi\\emoji-prediction\\glove.6B.50d.txt', encoding='utf-8')\n",
    "for line in f:\n",
    "    values = line.split()\n",
    "    word = values[0]\n",
    "    coefs = np.asarray(values[1:], dtype='float32')\n",
    "    embeddings_index[word] = coefs\n",
    "f.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Checking length of a particular word\n",
    "embeddings_index[\"i\"].shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from scipy import spatial\n",
    "# Checking cosine similarity of words happy and sad\n",
    "spatial.distance.cosine(embeddings_index[\"happy\"], embeddings_index[\"sad\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Checking cosine similarity of words India and Delhi\n",
    "spatial.distance.cosine(embeddings_index[\"india\"], embeddings_index[\"delhi\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Checking cosine similarity of words france and paris\n",
    "spatial.distance.cosine(embeddings_index[\"france\"], embeddings_index[\"paris\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Filling the embedding matrix\n",
    "embedding_matrix_train = np.zeros((X_train.shape[0], 10, 50))\n",
    "embedding_matrix_test = np.zeros((X_test.shape[0], 10, 50))\n",
    "\n",
    "for ix in range(X_train.shape[0]):\n",
    "    for ij in range(len(X_train[ix])):\n",
    "        embedding_matrix_train[ix][ij] = embeddings_index[X_train[ix][ij].lower()]\n",
    "        \n",
    "for ix in range(X_test.shape[0]):\n",
    "    for ij in range(len(X_test[ix])):\n",
    "        embedding_matrix_test[ix][ij] = embeddings_index[X_test[ix][ij].lower()]        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print (embedding_matrix_train.shape, embedding_matrix_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Training model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## - Using RNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# A simple RNN network to classify the emoji class from an input Sentence\n",
    "\n",
    "model = Sequential()\n",
    "model.add(SimpleRNN(64, input_shape=(10,50), return_sequences=True))\n",
    "model.add(Dropout(0.5))\n",
    "model.add(SimpleRNN(64, return_sequences=False))\n",
    "model.add(Dropout(0.5))\n",
    "model.add(Dense(5))\n",
    "model.add(Activation('softmax'))\n",
    "\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setting Loss and Optimiser for the model\n",
    "model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Training of the model and Setting hyperparameters for the model\n",
    "hist = model.fit(embedding_matrix_train,Y_train,\n",
    "                epochs = 50, batch_size=32,shuffle=True\n",
    "                )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prediction of the trained model \n",
    "pred = model.predict_classes(embedding_matrix_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### ACCURACY"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculating the accuracy of the algorithm\n",
    "float(sum(pred==Y_test))/embedding_matrix_test.shape[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Printing the sentences with the predicted and labled emoji\n",
    "for ix in range(embedding_matrix_test.shape[0]):\n",
    "    \n",
    "    if pred[ix] != Y_test[ix]:\n",
    "        print(ix)\n",
    "        print (test[0][ix],end=\" \")\n",
    "        print (emoji.emojize(emoji_dict[pred[ix]], use_aliases=True),end=\" \")\n",
    "        print (emoji.emojize(emoji_dict[Y_test[ix]], use_aliases=True))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Predicting for our random sentence\n",
    "x = ['i', 'do', 'think','this', 'class', 'is', 'very', 'interesting']\n",
    "\n",
    "x_ = np.zeros((1,10,50))\n",
    "\n",
    "for ix in range(len(x)):\n",
    "    x_[0][ix] = embeddings_index[x[ix].lower()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.predict_classes(x_)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##  - Using LSTM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# A simple LSTM network\n",
    "model = Sequential()\n",
    "model.add(LSTM(128, input_shape=(10,50), return_sequences=True))\n",
    "model.add(Dropout(0.5))\n",
    "model.add(LSTM(128, return_sequences=False))\n",
    "model.add(Dropout(0.5))\n",
    "model.add(Dense(5))\n",
    "model.add(Activation('softmax'))\n",
    "\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setting Loss ,Optimiser for model\n",
    "model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Training model\n",
    "hist = model.fit(embedding_matrix_train,Y_train,\n",
    "                epochs = 50, batch_size=32,shuffle=True\n",
    "                )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prediction of trained model\n",
    "pred = model.predict_classes(embedding_matrix_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### ACCURACY"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculating accuracy / score  of the model\n",
    "float(sum(pred==Y_test))/embedding_matrix_test.shape[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Printing the sentences with the predicted and the labelled emoji\n",
    "for ix in range(embedding_matrix_test.shape[0]):\n",
    "    \n",
    "    if pred[ix] != Y_test[ix]:\n",
    "        print(ix)\n",
    "        print (test[0][ix],end=\" \")\n",
    "        print (emoji.emojize(emoji_dict[pred[ix]], use_aliases=True),end=\" \")\n",
    "        print (emoji.emojize(emoji_dict[Y_test[ix]], use_aliases=True))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
