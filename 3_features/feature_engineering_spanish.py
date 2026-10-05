"""
Feature Engineering (streaming)
===============================

Doel
----
Dit bestand definieert alle Spaanse *streaming features* die we uit de WOK-tabellen
berekenen. Elke feature is een Python-functie die op een uniforme manier
aangeroepen wordt door de runner (`clean_and_parse.py`).

Uniforme signature
------------------
Elke feature-functie volgt dezelfde call-signature:

    stream_fn(
        tables: Dict[str, List[Path]],
        *,
        chunksize: int,
        log_path: Path | None,
        verbose: bool
    ) -> pd.DataFrame

- `tables`: dictionary met logische tabelnamen als keys en een lijst van CSV-paths
- De functie streamt zelf de bestanden via `iter_csv_chunks`
- Return: DataFrame met per speler één featurekolom, altijd inclusief:
    ["Player_Profile_ID", "<feature_name>"]

FEATURES_REGISTRY
-----------------
Onderin dit bestand staat `FEATURES_REGISTRY`: de "bron van waarheid".
Daar staat per feature:
- `stream_fn`: de Python-functie
- `tables`: welke tabellen nodig zijn
- `usecols`: welke kolommen minimaal geladen moeten worden
- `log_name`: naam van het logbestand
- `kwargs`: extra parameters voor de functie

De runner gebruikt alléén deze registry. Hij weet niets van de inhoud van de features.

Nieuwe feature toevoegen — stappenplan
--------------------------------------
1. Schrijf de stream_fn
   - Definieer een nieuwe functie met de uniforme signature.  
   - Stream de juiste tabellen uit `tables`.  
   - Bereken de feature per speler.  
   - Return een DataFrame met twee kolommen:
        ["Player_Profile_ID", "<jouw_feature_naam>"]

2. Voeg hem toe aan FEATURES_REGISTRY
   - Kies een unieke key (bijv. `"std_casino"`)  
   - Zet daar je functie, benodigde tabellen/kolommen en logbestand in.  

3.	Activeer de feature in een scenario
	- In clean_and_parse.py staat de dictionary SCENARIOS.
	- Voeg je feature-key toe aan de lijst features van een bestaand scenario, of maak een nieuw scenario aan.

Tijdsfiltering voor Spaanse features (X-periode filtering)
-----------------------------------------------------------
Voor het Spaanse scenario willen we features berekenen over een flexibele X-periode
en targets over een Y-periode. Dit werkt als volgt:

**Hoe werkt de flow?**

1. **SBATCH script** roept parse_runner.py aan met environment variables
   (OPERATOR_PREFIX, X_TIJDSPAD, Y_TIJDSPAD, SCENARIO).

2. **parse_runner.py** (regels 118-126) haalt X_TIJDSPAD en Y_TIJDSPAD uit env vars
   en geeft ze door als command line arguments aan clean_and_parse.py:
   ```python
   x_tijdspad = os.getenv("X_TIJDSPAD")  # "01012025:31012025"
   if x_tijdspad:
       cmd.extend(["--x-tijdspad", x_tijdspad])
   ```

3. **clean_and_parse.py** parsed deze arguments en converteert ze naar lists:
   ```python
   x_tijdspad = ["01012025", "31012025"]  # van colon-separated naar list
   y_tijdspad = ["01022025", "21022025"]
   ```

4. **clean_and_parse.py** voegt deze toe aan kwargs voor de features in het scenario:
   ```python
   # Voor X-features (f0, f1, f3):
   kwargs["x_tijdspad"] = x_tijdspad

   # Voor Y-target (maak_self_exclusion_flexible_y):
   kwargs["y_tijdspad"] = y_tijdspad
   ```

5. **Features ontvangen x_tijdspad via kwargs** en filteren data:
   ```python
   def f0_net_winloss(
       tables: Dict[str, List[Path]],
       *,
       x_tijdspad: List[str] | None = None,  # ['01012025', '31012025']
       chunksize: int = 200_000,
       ...
   ):
       # Converteer naar timestamps
       start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
       eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])

       # Stream data en filter op periode
       for df in iter_csv_chunks(...):
           ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True)
           mask_periode = (ts >= start_datum) & (ts < eind_datum)
           df = df[mask_periode]  # Filter alleen data in X-periode
           # ... rest van processing
   ```

**Belangrijke patronen (zie feature_engineering.py: maak_flexible_x_features):**

- **Meestal één pass** voor Spaanse features; F26–F28 delen een extra pass voor saldoreconstructie
  - Spaanse features: filter direct binnen de loop, return alleen spelers met data
  - Nederlandse features: Pass 1 verzamelt ALLE spelers, Pass 2 filtert, return ALL met NaN

- **Tijdsfiltering binnen de chunk loop:**
  ```python
  for df in iter_csv_chunks(paths=tx_paths, usecols=..., chunksize=...):
      # 1. Parse timestamps
      ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)

      # 2. Filter op periode (start inclusief, eind exclusief)
      mask_periode = (ts >= start_datum) & (ts < eind_datum)
      if not mask_periode.any():
          continue  # Skip chunk als geen data in periode

      # 3. Gebruik gefilterde data
      df = df.loc[mask_periode]
      # ... rest van feature berekening
  ```

- **usecols moet datetime kolommen bevatten** voor filtering:
  - WOK_Bet: "Bet_Start_Datetime"
  - WOK_Game_Session: "Game_Session_Start_Datetime"
  - WOK_Player_Account_Transaction: "Transaction_Datetime"

**Scenario definitie (in clean_and_parse.py):**

```python
"Flexible_two_months_spanish": {
    "features": [
        "f0_net_winloss",           # X-features met tijdsfiltering
        "f1_active_days",
        "f3_total_wagered",
        "maak_self_exclusion_flexible_y"  # Y-target (uit feature_engineering.py)
    ]
}
```

**Registry update (kwargs blijven leeg, worden runtime gevuld):**

De FEATURES_REGISTRY hoeft GEEN default x_tijdspad te bevatten - dit wordt
runtime doorgegeven via clean_and_parse.py. De kwargs in registry blijven leeg:

```python
"f0_net_winloss": {
    "stream_fn": f0_net_winloss,
    "tables": ["WOK_Player_Account_Transaction"],
    "usecols": {
        "WOK_Player_Account_Transaction": [
            "Player_Profile_ID",
            "Transaction_Amount",
            "Transaction_Datetime",  # <-- Nodig voor filtering!
            "Transaction_Type",
            "Transaction_Status",
        ],
    },
    "log_name": "f0_net_winloss.log",
    "kwargs": {},  # Blijft leeg, x_tijdspad komt van clean_and_parse.py
},
```

Hier maken we de Spaanse features:

Gambling Intensity & Outcomes
  • Net Win/Loss (F0) – Calculate the player’s net gambling result over the period. Use WOK_Player_Account_Transaction records with Transaction_Type = STAKE, WINNING, CASH_OUT, VOID_BET, VOID_STAKE, BONUS. Sum all transaction amounts (stakes are negative, winnings and returned stakes are positive). This sum is the net gain or loss.
  • Active Days (F1) – Count the number of distinct calendar days in the period on which the player had any gambling activity. Use timestamps from WOK_Bet (bet placement time) and WOK_Game_Session (session start time) to identify days with at least one interaction. (Output is an integer ≥ 2)
  • Net Loss per Day (F2) – Compute the average net loss per active day. First derive net win/loss (as in F0), then divide by F1 (days active). (If positive net, this could be net win per day)
  • Total Wagered (F3) – Calculate total money participated (staked) by the player. Sum all amounts for Transaction_Type = STAKE (money wagered, treated as negative) and add any refunded stakes (e.g. VOID_BET/VOID_STAKE transactions, positive amounts). The result will be a negative number (total money staked minus any returned).
  • Average Wager per Day (F4) – Compute the average amount wagered per active day. Divide F3 (total wagered, as a negative value) by F1 (active days). This yields a negative number (or zero) representing mean daily spend.
  • RTP Deviation (F7) – Determine the player’s return-to-player ratio and its deviation from 0.95. Calculate actual RTP = (Total winnings received) / (–F3 total staked). Sum all Transaction_Type = WINNING amounts and divide by the absolute total staked (from F3). Then take the absolute difference between this ratio and 0.95.
  • Interactions per Day (F8) – Calculate the average number of gambling interactions per day. First count total interactions in the period: each placed bet (WOK_Bet) and each game session (WOK_Game_Session) counts as one interaction. Then divide by F1 (active days).
  • Big Wins per Day (F9) – Compute the average count of significant wins per day. Count the number of winning outcomes where the payout was greater than the stake (e.g. a bet that returned more money than was wagered – essentially any win, excluding purely refunded stakes). Use WOK_Bet (if it provides stake and payout info) or compare WOK_Player_Account_Transaction entries (WINNING vs corresponding STAKE) to identify such wins. Divide that count by F1.

Player Profile Demographics
  • Sex (F5) – Record the player’s gender as a binary value. If available from the player profile (e.g. WOK_Player_Profile data), set 1 for male and 0 for female. (If gender is not provided in CDB data, this variable is infeasible.)
  • Age (F6) – Compute the player’s age at the end of the period. Use date of birth from WOK_Player_Profile (Player_Profile_DOB) and subtract from the end-date of analysis (in years, floor to integer).

Deposit & Withdrawal Patterns
  • Canceled Withdrawals per Day (F10) – Calculate the average number of withdrawal attempts canceled by the player per day. In WOK_Player_Account_Transaction, count transactions of type WITHDRAWAL that were not completed (e.g. Transaction_Status = UNSUCCESSFUL, indicating the withdrawal was reversed or failed). Divide by F1. (If the data does not explicitly log user-initiated cancellation, use failed withdrawal attempts as an approximation.)
  • Withdrawals per Day (F11) – Calculate average completed withdrawals per day. Count WOK_Player_Account_Transaction records with Transaction_Type = WITHDRAWAL (successful only) and divide by F1.
  • Deposits per Day (F12) – Calculate average deposit count per day. Count WOK_Player_Account_Transaction records with Transaction_Type = DEPOSIT (successful deposits) and divide by F1.
  • Canceled Deposits per Day (F13) – Calculate average number of failed or canceled deposit attempts per day. Count WOK_Player_Account_Transaction entries with Transaction_Type = DEPOSIT that have Transaction_Status = UNSUCCESSFUL (e.g. declined or aborted deposits) and divide by F1.
  • Payment Method Variety (F24) – Determine the number of different payment instruments the player used for transactions in the period. Use the Transaction_Deposit_Instrument field in WOK_Player_Account_Transaction for deposits (e.g. CREDIT_CARD, BANK_TRANSFER, etc.) and, if applicable, any instrument info for withdrawals. Count the distinct payment types used in all deposit/withdrawal transactions.
  • Deposit Amount Variability (F60) – Compute the variability of deposit sizes. Take all deposit transactions (from WOK_Player_Account_Transaction with type DEPOSIT) and calculate the coefficient of variation (standard deviation divided by mean) of their amounts. (Higher values indicate inconsistent deposit amounts.)

Limits and Self-Exclusion
  • Limit Increases (F22) – Count how many times the player raised their play limits during the period. Use the WOK_Player_Limits table, focusing on deposit limits and/or wagering (participation) limits. Each time a new limit record comes into effect, compare it to the previous value for the same limit type – if the limit amount increased, count it. (If multiple limits are raised simultaneously, count as one increase event.)
  • Limit Decreases (F23) – Count how many times the player lowered their play limits during the period. Similarly, use WOK_Player_Limits records and count each instance a daily/weekly/monthly limit value is decreased relative to the prior value. (If limits in different categories are decreased at the same time, each is counted.)
  • Self-Exclusions (F25) – Count the number of times the player has self-excluded over their entire play history. (Likely infeasible: self-exclusion events are not recorded in the CDB data model, since Dutch self-exclusions are handled via the separate CRUKS system. Operators’ internal records of self-exclusion requests are not part of the provided CSV tables.)

Risk Behaviors (Chasing Losses/Wins)
  • Balance Drop Events (F26) – Count how many times the player’s account balance fell from above €2 to below €2 during the period. To calculate, reconstruct the account balance chronologically using all transactions. Identify instances where, after a bet or other debit, the balance goes below €2 when it was ≥ €2 before. Count each such threshold-crossing event.
  • Deposits After Low Balance (F27) – Count how many times the player made a deposit shortly after depleting their balance below €2. For each event identified in F26 (balance < €2), check if the player’s next transaction was a deposit. Count the number of deposits that were preceded by a sub-€2 balance event. Divide by F1 to normalize per day. (If the player never dropped below €2, this can be “NA”.)
  • Time to Replenish Low Balance (F28) – Measure the median time (in seconds) it takes for the player to deposit after their balance falls below €2. For each balance drop event (from F26), calculate the time difference until the next deposit transaction. Then take the median of these intervals. (If no deposits followed low-balance events, output NA.)
  • Quick Re-Bet After Loss (F51) – Calculate the typical time the player waits after losing a bet before placing a new bet. For each resolved bet that returned no winnings (a loss), find the timestamp of the next bet placement. Compute all such intervals and take the median (in seconds). (Requires identifying lost bets: e.g. a bet with a STAKE but no corresponding WINNING transaction; assume WOK_Bet can indicate when a bet is settled and whether it was lost.)
  • Larger Bets After Big Win (F52) – Count instances of the player substantially increasing their stake sizes following a major win. To do this, scan the sequence of bets: whenever the player receives a large win (significant payout), compare the average stake of the next ~10 bets to the average stake of the previous ~10 bets. If the post-win average bet size is at least double the pre-win average, count this as an occurrence. (Use at least 5 bets before/after if 10 are not available. If fewer than 5 prior or subsequent bets, that win is not evaluated.) Sum all such occurrences in the period.

Game Preferences & Diversity
  • Bet Countries Count (F17) – Likely infeasible: Intended to be the number of different countries in which the sports/events bet on took place. The CDB does not contain structured data about event locations. (Would require external mapping of each bet’s event to a country, which isn’t available in the CSV data.)
  • Sports Types Count (F18) – Likely infeasible: Number of different sports or betting categories the player wagered on. The CDB WOK_Bet data does not explicitly label the sport of each event in a structured field. (Determining this would require parsing event descriptions or an external sports mapping.)
  • Competition Count (F19) – Likely infeasible: Number of different competitions or leagues bet on. As with F18, the data safe doesn’t provide a field for competition names, so this cannot be reliably derived from the CSV alone.
  • Domestic vs International Bet % (F20) – Likely infeasible: Percentage of bets placed on events in Spain or international competitions. This requires knowing which bets correspond to Spanish events, which is not included in CDB data.
  • Non-EUR Currency Interactions (F21) – Likely infeasible: Number of gambling interactions made in a currency other than Euros (divided by F1). The Dutch CDB records all stakes and transactions in EUR (converting other currencies to EUR value), and it does not flag the original currency. Assume 0 or not applicable in this context.
  • Game Types Count (F32) – Count the number of different game types or genres the player participated in during the period. Use both WOK_Bet and WOK_Game_Session data. For sports bets (WOK_Bet), consider “sports betting” as one category. For casino sessions (WOK_Game_Session), determine the game type via the game identifier (join with WOK_Game to get the category, e.g. roulette, slots, poker, etc.). Count the distinct categories of games the player played (e.g. Sports bet, Slots (casino games of chance), Roulette, Poker, Blackjack, Bingo, etc.).
  • Segment APU – Sports Betting 70% Flag (F33) – Determine if the player’s play is predominantly in the betting segment. Calculate the total amount staked on sports/horse race bets (all stakes from WOK_Bet) and divide by the player’s total staked amount (sum of all stakes). If ≥ 0.70 (70%), set this indicator to 1, otherwise 0.
  • Segment AZA – Casino Games Flag (F34) – Determine if at least 70% of total stakes are in games of chance (casino-style games, excluding those covered by specific categories below). Sum all stakes placed in general casino game sessions (e.g. slot machines or other RNG games in WOK_Game_Session that are not roulette, poker, blackjack, or bingo) and check if this constitutes ≥ 70% of total stakes. If yes, F34 = 1, else 0.
  • Segment RLT – Roulette Flag (F35) – Determine if ≥ 70% of stakes are on roulette. Sum stakes in roulette sessions (WOK_Game_Session where game type is roulette) and check fraction of total stakes. Set to 1 if ≥ 0.70, else 0.
  • Segment POQ – Poker Flag (F36) – Determine if ≥ 70% of stakes are on poker. Sum stakes in poker games (e.g. poker tournaments or cash games, if recorded in sessions or bets) and check if that’s ≥ 70% of total. Set flag accordingly.
  • Segment BLJ – Blackjack Flag (F37) – Determine if ≥ 70% of stakes are on blackjack. Sum stakes in blackjack sessions and check fraction of total. 1 if ≥ 0.70, otherwise 0.
  • Segment BNG – Bingo Flag (F38) – Determine if ≥ 70% of stakes are on bingo. Sum stakes in bingo sessions and check fraction. 1 if ≥ 0.70, else 0.
  • Dominant Segment Share (F39) – Calculate the proportion of the player’s stakes concentrated in their top game segment. Compute total stakes for each segment (sports, slots/azar, roulette, poker, blackjack, bingo as above) and identify the maximum. Then compute that maximum divided by the total staked across all games. This yields a number between 0 and 1 representing the share of play in the dominant category.
  • Unique Games per Day (F40) – Calculate the average number of distinct game titles played per day. Use WOK_Game_Session data to find unique game identifiers (Game_ID or names) the player engaged with during the period. Count the number of distinct games played in the period, then divide by F1 (days active). (If the player did not play any game sessions, this can be NA or 0.)

Time-of-Day Play Patterns
  • Heavy-Play Hours Count (F41) – Identify the number of different hour-of-day slots in which the player concentrated their activity. For each hour of the day (00:00–00:59, 1:00–1:59, … 23:00–23:59), calculate the total number of interactions (bets or session starts) that occurred in that hour across the entire period. Determine which hours account for at least 2.083% of the player’s total interactions (2.083% = 1/48, representing a notable concentration of activity in that hour). Count how many distinct hours meet or exceed this threshold. (Result is between 0 and 24.)
  • Morning Interaction % (F42) – Calculate the proportion of the player’s interactions that occur during the morning/early afternoon window (08:00:00–15:59:59). Count all interactions (bets placed or sessions started) within that daily time block and divide by the total number of interactions in the period. Output is a fraction between 0 and 1 (or 0 if no play in that window).
  • Evening Interaction % (F43) – Calculate the proportion of interactions that occur in the late afternoon/evening window (16:00:00–23:59:59). Similarly, count interactions in that time range and divide by total interactions.
  • Morning Stakes % (F44) – Calculate the share of money wagered in the morning window. Sum all stake amounts (Transaction_Type = STAKE in WOK_Player_Account_Transaction) for bets placed between 08:00 and 15:59, and divide by the total amount staked (sum of all STAKE transactions) in the period. Result is a fraction 0–1.
  • Evening Stakes % (F45) – Calculate the share of money wagered in the evening window (16:00–23:59). Sum all stake amounts for interactions in that time frame and divide by total stakes.

Session & Duration Metrics
  • Sessions per Day (F29) – Compute the average number of casino game sessions per day. Count the total number of WOK_Game_Session records for the player in the period (each session of a slot, roulette, blackjack, etc. is one count). Divide this count by F1 (days active). (If the player had no game sessions, this can be 0.)
  • Concurrent Play Avg (F30) – Measure the player’s tendency to multi-play during casino sessions. For each WOK_Game_Session (each session of “other games”), count how many other interactions (bets on different games or other sessions) the player started during that session’s timeframe. Compute the average of these counts across all sessions. (If the player never played multiple games at once, this will be 0.)
  • Median Rounds per Session (F31) – Read Game_Session_Rounds from WOK_Game_Session and take the median of the valid round counts per player. Use WOK_Game_Session_Transaction only to link sessions to players; each session contributes once per player regardless of its transaction count. Missing or invalid round counts are not replaced with transaction counts.
  • Median Bet Resolution Time (F46) – Determine the median time (in seconds) between placing a bet and its resolution. For all bets in WOK_Bet that have a settlement (outcome), calculate the time difference between the bet acceptance timestamp and the bet settlement/result time. Take the median of these durations. (Requires that WOK_Bet or related data provides both the bet placement time and the time the bet was settled. Exclude bets that were not resolved in the period.)
  • Median Session Length (F47) – Compute the median duration of game sessions. Use WOK_Game_Session start and end times to find the length of each session (in seconds). Then take the median of all session lengths. (If the player had no game sessions, this can be NA.)

Betting Behavior & Features
  • Cash-Out Usage % (F48) – Calculate the percentage of bets that the player cashed out early. Use WOK_Player_Account_Transaction to identify cash-out events (Transaction_Type = CASH_OUT). Count the number of distinct bets that had a cash-out, and divide by the total number of bets placed (WOK_Bet count). (If multiple partial cashouts on one bet are possible, ensure each bet is only counted once.)
  • Live Bet % (F49) - Percentage of bets that were placed live (in-play). This would require knowing for each bet whether it was placed after the game/event started. The CDB does not explicitly flag in-play bets or provide event start times, so determining this from the data is not reliable.
  • Single Bet % (F50) – Determine the share of bets that were single bets (vs combination/multiple bets). Use the bet type field in WOK_Bet (Bet_Type) – count bets labeled as SINGLE and divide by the total number of bets. Output is a fraction (0–1). This could also be done with the bet_type field if it indicates single vs. combination bets.
  • Bet Odds Variability (F61) – Measure the variation in odds of the player’s bets. For all sports bets in the period, take the odds (quotation) of each bet (e.g. Part_Odds in WOK_Bet). Compute the coefficient of variation of these odds (standard deviation / mean of odds). (If the player made no bets, or if odds data is unavailable, this can be NA.)

Temporal Patterns & Trends
  • Active Period Span (F14) – Calculate the number of days between the player’s first and last monetary activity in the period. Find the date of the earliest transaction and the date of the latest transaction for that player (within the analysis window). Compute the difference in days and add 1 (inclusive). (Result is between 1 and ~183 days for a 6-month period.)
  • Active Day Fraction (F15) – Calculate the fraction of days in the span that the player was active. Divide F1 (active days count) by F14 (span of days from first to last play). This yields a value from 0 to 1 indicating how regularly the player engaged within the period.
  • Account Age (F16) – Calculate the calendar days elapsed between the first successful financial transaction in the available account history and the last successful financial transaction within the period. Financial transactions are STAKE, DEPOSIT and WITHDRAWAL. The result is at least 1 day. Historical transactions before the period are needed to determine financial activation; registration date is not used.
  • Stake Slope (First vs Second Half) (F53) – Measure the change in betting volume between the first half and second half of the period. First, find the median timestamp of the period (mid-point in time). Split all stakes into two groups: those placed before the median time and those after. Sum the stake amounts in each half. Compute the difference (second-half sum minus first-half sum), take the absolute value, and then divide by the total number of interactions in the period. This result is a normalized absolute “slope” of wagering activity over time.
  • Post-Median Active Days % (F54) – Calculate what portion of active days occurred in the second half of the period. Split the active days list by the median date of the period. Count how many of the player’s active days fall in the latter half, and divide by F1 (total active days). This yields a fraction (0–1) indicating if activity was skewed towards the end of the period.
  • Stake Variance Difference (F55) – Compare the variability of stake amounts between the first and second half of the period. Take all bet stakes from the first half (before median date) and compute their variance, and do the same for stakes in the second half. Then compute the absolute difference between the two variances.
  • Stake CV Difference (F56) – Compare the coefficient of variation of stakes between the first vs. second half. Calculate the coefficient of variation (std dev / mean) of stake sizes in the first half and in the second half. Then find the difference (second-half CV minus first-half CV). (This can be a positive or negative value, indicating increase or decrease in stake variability.)
  • Longest Daily Streak (F57) – Determine the player’s longest run of consecutive days with activity. Use the sorted list of active dates and find the largest sequence of back-to-back days with no gaps. Output the length of this longest streak (in days).
  • Longest Streak Ratio (F58) – Compute the longest streak as a fraction of total active days. Take the streak length from F57 and divide by F1. This yields a value between 0 and 1 showing how significant the longest run was relative to overall play frequency.
  • Median Break Duration (F59) – Calculate the typical length of breaks (inactivity periods) between play days. First, find all gaps between consecutive active days (in hours). For example, if a player played on Monday and next on Thursday, the break between those sessions is from end of Monday to start of Thursday (roughly 2 days gap). Compute all such inter-session or inter-day break lengths in hours. Exclude outliers using the 1.5×IQR rule (to avoid extreme breaks skewing the data). Then take the median of the remaining break durations. (If the player was active nearly every day or only had one session, this metric may be NA or 0.)

Gebruik Panda's build in functies met vectorisatie waar mogelijk voor performance.
  
We maken altijd eerst een test
tests_spanish_features

En checken de voorbeelden van de data hier:
0_example_csv_file

run de test
also run parse_runner as a test with the right scenario every time you've added a feature, to check if it still works

I made an example parsing function for how to get data out of json for each file type in feature_engineering.py.
WOK_Player_Profile: nr_of_bank_accounts
WOK_Player_Flags: nr_of_risk_classes
WOK_Player_Limits: latest_limits
WOK_Game_Session: avg_nr_of_game_transactions
WOK_Bet: avg_nr_of_bet_parts_per_bet
WOK_Complaint: nr_of_complaints_and_variance_in_nr_responses_per_complaint
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from path_finding import iter_csv_chunks
from balance_reconstruction import reconstruct_start_balances
from reading_difficult_json import simple_Player_Profile_Bank_Account_json_iterator, simple_RG_Class_Value_from_FLAG_RG_CLASS_json_iterator 
from mapping_helpers import build_txid_to_player_map_ram, haal_uit_bank_json_iterator
from reading_difficult_json import iter_limit_values, iter_transaction_ids_from_Game_Transactions, iter_part_ids_from_Bet_Parts, iter_player_profile_ids_from_Bet_Transactions, iter_transaction_ids_from_Bet_Transactions, get_list_of_response_ids_from_Responses_list, iter_part_live_flags_from_Bet_Parts, _safe_load_json_relaxed
# organisatie-relationeel: joins op de losse tabellen (vervangen de JSON-readers hierboven)
from relational_joins import (
    build_session_tx_map, build_bet_tx_map, build_bet_parts_map, build_limits_events_map,
    build_bet_players_map,
)

def _setup_feature_logger(log_path: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(f"feature:{name}")
    for _h in logger.handlers[:]:   # sluit oude handler écht (anders fd-lek bij hergebruik per featurenaam)
        _h.close()
        logger.removeHandler(_h)
    logger.setLevel(logging.INFO)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger

# ------------------------------
# Helper functies voor tijdsfiltering
# ------------------------------

def parse_ddmmyyyy_to_timestamp(date_str: str) -> pd.Timestamp:
    """
    Converteer 'DDMMYYYY' string naar pd.Timestamp.

    Voorbeeld: '01012025' -> pd.Timestamp('2025-01-01')
    """
    day = date_str[:2]
    month = date_str[2:4]
    year = date_str[4:]
    return pd.Timestamp(f"{year}-{month}-{day}")


def filter_dataframe_by_timeperiod(
    df: pd.DataFrame,
    datetime_column: str,
    x_tijdspad: List[str] | None,
) -> pd.DataFrame:
    """
    Filter een DataFrame op basis van tijdsperiode.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame om te filteren
    datetime_column : str
        Naam van de datum/tijd kolom
    x_tijdspad : List[str] | None
        [start_datum, eind_datum] in formaat 'DDMMYYYY'
        Als None, geen filtering

    Returns
    -------
    pd.DataFrame
        Gefilterde DataFrame
    """
    if x_tijdspad is None:
        return df

    # Parse tijdspad
    start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
    eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])

    # Converteer kolom naar datetime
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors="coerce", utc=True)

    # Filter op periode (start inclusief, eind exclusief zoals gebruikelijk)
    mask = (df[datetime_column] >= start_datum) & (df[datetime_column] < eind_datum)

    return df[mask].copy()

# ------------------------------
# F0: Net Win/Loss - Netto winst/verlies
# ------------------------------

def f0_net_winloss(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F0: Net Win/Loss

    Bereken de netto winst/verlies van de speler over de periode.
    Resultaat = (Winsten + Bonussen) - (Inzetten + Verlopen Bonussen)

    Input:
        - WOK_Player_Account_Transaction

    Parameters:
        x_tijdspad: List[str] | None
            [start_datum, eind_datum] in formaat 'DDMMYYYY'

    Output:
        - f0_net_winloss: Float (positief = winst voor speler, negatief = verlies voor speler)
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f0_net_winloss")
        logger.info("▶ START F0: Net Win/Loss")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]}")
    else:
        logger = None

    # Parse tijdspad indien gegeven
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    # Uitgebreide set transactietypes
    relevant_types = {
        # POSITIEF (Speler krijgt geld)
        'WINNING',
        'CASH_OUT',
        'VOID_BET',
        'VOID_STAKE',
        'BONUS',
        
        # NEGATIEF (Speler betaalt/verliest geld - bedrag moet negatief zijn in data)
        'STAKE',
        'BONUS_CANCELLED',  # Toegevoegd: Bonus ingetrokken
        'BONUS_EXPIRED',    # Toegevoegd: Bonus verlopen
        
        # VARIABEL (Correcties, kan pos of neg zijn)
        'RESETTLEMENT'      # Toegevoegd: Correcties op eerdere settlement
    }

    # Verzamel net per speler
    net_per_speler: Dict[str, float] = {}

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f0_net_winloss"])

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Amount", "Transaction_Datetime", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        # Filter NaN player IDs
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        # Filter op tijdsperiode
        if start_datum is not None:
            ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()

        # Filter succesvolle transacties van relevante types
        df = df[df["Transaction_Status"] == "SUCCESSFUL"]
        df = df[df["Transaction_Type"].isin(relevant_types)]

        if df.empty:
            continue

        # Converteer amounts naar numeriek
        df["amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        df = df[df["amount"].notna()]

        # -----------------------------------------------------------
        # OPTIONELE VEILIGHEIDSCHECK (uncomment indien data tekens mist)
        # -----------------------------------------------------------
        # Als je data soms positieve getallen heeft voor 'STAKE' of 'BONUS_EXPIRED',
        # forceer ze hier naar negatief:
        #
        # negative_types = {'STAKE', 'BONUS_CANCELLED', 'BONUS_EXPIRED'}
        # mask_neg = df["Transaction_Type"].isin(negative_types)
        # df.loc[mask_neg, "amount"] = -df.loc[mask_neg, "amount"].abs()
        # -----------------------------------------------------------

        # Groepeer per speler en som
        # (Positieve bedragen tellen op, negatieve bedragen trekken af)
        speler_sums = df.groupby("Player_Profile_ID")["amount"].sum()

        for player_id, amount in speler_sums.items():
            net_per_speler[player_id] = net_per_speler.get(player_id, 0.0) + float(amount)

    # Bouw output
    records = [
        {"Player_Profile_ID": pid, "f0_net_winloss": net}
        for pid, net in net_per_speler.items()
    ]

    result = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ F0 Net Win/Loss klaar: {len(result):,} spelers")
        if len(result) > 0:
            logger.info(f"   Gemiddeld net: {result['f0_net_winloss'].mean():.2f}")

    return result

# ------------------------------
# F1: Active Days - Aantal unieke dagen met activiteit
# ------------------------------

def f1_active_days(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    print("▶ START F1: Active Days (via Transactions)")
    print("▶ START F1: Active Days (via Transactions)")
    print("▶ START F1: Active Days (via Transactions)")
    """
    F1: Active Days (Geoptimaliseerd)

    Tel het aantal unieke dagen waarop de speler actief was door te kijken naar
    inzet-transacties (STAKE) in de rekeningmutaties.

    Input:
        - WOK_Player_Account_Transaction: Transaction_Type == 'STAKE'

    Parameters:
        x_tijdspad: List[str] | None
            [start_datum, eind_datum] in formaat 'DDMMYYYY'

    Output:
        - f1_active_days: Integer (aantal unieke dagen met inzet)
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f1_active_days")
        logger.info("▶ START F1: Active Days (via Transactions)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]}")
    else:
        logger = None

    # Parse tijdspad
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None
    
    print(f"Tables: {tables.keys()}")
    # Set om unieke (Player, Date) tupels op te slaan
    # We gebruiken tuples omdat dit geheugenefficiënter is dan dicts met sets voor enorme datasets
    player_dates = set()

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        if verbose:
            print("⚠️ Geen transactiedata gevonden, F1 Active Days kan niet berekenen")
        return pd.DataFrame(columns=["Player_Profile_ID", "f1_active_days"])

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Type"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        # 1. Filter lege spelers
        if verbose:
            print(f"[f1] Chunk geladen: {len(df):,} records")
        df = df[df["Player_Profile_ID"].notna()]
        if verbose:
            print(f"[f1] Na filter niet-lege Player_Profile_IDs: {len(df):,} records")
        if df.empty:
            continue

        # 2. Filter alleen op 'STAKE' (Dit impliceert actieve inzet/deelname)
        df = df[df["Transaction_Type"] == "STAKE"]
        if verbose:
            print(f"[f1] Na filter STAKE: {len(df):,} records")
        if df.empty:
            continue

        # 3. Tijd conversie en filtering
        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        
        if start_datum is not None:
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode]
            ts = ts.loc[mask_periode]
            if verbose:
                print(f"[f1] Na tijdsfiltering: {len(df):,} records")

        # 4. Voeg unieke (Player, Date) combinaties toe aan de set
        # We itereren over de zip om het snel in de set te proppen
        current_dates = ts.dt.date
        current_pids = df["Player_Profile_ID"]
        
        if verbose:
            print(f"[f1] Unieke spelers in chunk: {current_pids.nunique():,}, unieke dagen: {current_dates.nunique():,}")
        for pid, d in zip(current_pids, current_dates):
            player_dates.add((pid, d))
        if verbose:
            print(f"[f1] Tot nu toe unieke (Player, Date) combinaties: {len(player_dates):,}")

    # 5. Aggregeren: Tel unieke dates per speler
    # We bouwen een teller op vanuit de unieke set
    counts: Dict[str, int] = {}
    for pid, _ in player_dates:
        counts[pid] = counts.get(pid, 0) + 1
    if verbose:
        print(f"[f1] Unieke spelers met activiteit: {len(counts):,}")

    # 6. Output bouwen
    records = [
        {"Player_Profile_ID": pid, "f1_active_days": count}
        for pid, count in counts.items()
    ]
    
    result = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ F1 Active Days klaar: {len(result):,} spelers")
        if len(result) > 0:
            logger.info(f"   Gemiddeld: {result['f1_active_days'].mean():.1f} dagen")

    return result

# ------------------------------
# F2: Net Loss per Day - Gemiddeld netto verlies per actieve dag
# ------------------------------

def f2_net_loss_per_day(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F2: Net Loss per Day

    Bereken het gemiddeld netto verlies per actieve dag.
    Dit is F0 (Net Win/Loss) gedeeld door F1 (Active Days).

    Output:
        - f2_net_loss_per_day: Float (positief = winst per dag, negatief = verlies per dag)
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f2_net_loss_per_day")
        logger.info("▶ START F2: Net Loss per Day")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]}")
    else:
        logger = None

    # Bereken F0 (net win/loss)
    f0_result = f0_net_winloss(tables, x_tijdspad=x_tijdspad, chunksize=chunksize, log_path=None, verbose=verbose)

    # Bereken F1 (active days)
    f1_result = f1_active_days(tables, x_tijdspad=x_tijdspad, chunksize=chunksize, log_path=None, verbose=verbose)

    # Merge op Player_Profile_ID
    if f0_result.empty or f1_result.empty:
        if logger:
            logger.warning("⚠️ F0 of F1 is leeg, kan F2 niet berekenen")
        return pd.DataFrame(columns=["Player_Profile_ID", "f2_net_loss_per_day"])

    merged = f0_result.merge(f1_result, on="Player_Profile_ID", how="inner")

    # Bereken F2: net / active_days (vectorized)
    merged["f2_net_loss_per_day"] = np.where(
        merged["f1_active_days"] > 0,
        merged["f0_net_winloss"] / merged["f1_active_days"],
        np.nan
    )

    result = merged[["Player_Profile_ID", "f2_net_loss_per_day"]].copy()

    if logger:
        logger.info(f"✅ F2 Net Loss per Day klaar: {len(result):,} spelers")
        if len(result) > 0:
            logger.info(f"   Gemiddeld: {result['f2_net_loss_per_day'].mean():.2f} per dag")

    return result

# ------------------------------
# F3: Total Wagered - Totaal ingezet bedrag
# ------------------------------
# One-time warning flag (module-level)
SIGN_WARNING_EMITTED = False

def f3_total_wagered(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F3: Participation / Total wagered (net of voids), computed from
    WOK_Player_Account_Transaction:

      - STAKE      : should be negative
      - VOID_BET   : should be positive
      - VOID_STAKE : should be positive

    F3 = sum(STAKE + VOID_BET + VOID_STAKE) per Player_Profile_ID.

    Notes:
      - We do NOT apply any "+1 day" logic to the date range.
      - Date filtering uses a half-open interval: [start, end)
        i.e. ts >= start and ts < end.

    Parameters
    ----------
    tables : Dict[str, List[Path]]
        Must contain key "WOK_Player_Account_Transaction" with one or more CSV paths.
    x_tijdspad : List[str] | None
        [start_datum, eind_datum] in 'DDMMYYYY'. If None, no time filtering.
        End is exclusive due to [start, end) filtering.
    chunksize : int
        CSV chunk size for streaming reads.
    log_path : Path | None
        If provided, uses your _setup_feature_logger(log_path, ...) helper.
    verbose : bool
        If True, prints one-time warning when coercion happens and no logger exists.

    Returns
    -------
    pd.DataFrame
        Columns: ["Player_Profile_ID", "f3_total_wagered"]
    """
    # Optional logger (assumes your helper exists in the codebase)
    if log_path:
        logger = _setup_feature_logger(log_path, "f3_total_wagered")
        logger.info("▶ START F3: Participation (STAKE + VOID_BET + VOID_STAKE)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering (half-open): {x_tijdspad[0]} - {x_tijdspad[1]} (end exclusive)")
    else:
        logger = None

    # Parse time window
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f3_total_wagered"])

    # Use a Series accumulator for speed
    acc = pd.Series(dtype="float64")  # index=Player_Profile_ID, values=sum(amount)

    relevant_types = {"STAKE", "VOID_BET", "VOID_STAKE"}

    global SIGN_WARNING_EMITTED

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=[
            "Player_Profile_ID",
            "Transaction_Amount",
            "Transaction_Datetime",
            "Transaction_Type",
            "Transaction_Status",
        ],
        chunksize=chunksize,
        verbose=verbose,
    ):
        # Drop missing players
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        # Time filter: [start, end) WITHOUT any "+1 day" logic
        if start_datum is not None:
            ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask = (ts >= start_datum) & (ts < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        # Successful only
        df = df[df["Transaction_Status"] == "SUCCESSFUL"]
        if df.empty:
            continue

        # Relevant transaction types only
        df = df[df["Transaction_Type"].isin(relevant_types)]
        if df.empty:
            continue

        # Numeric amount
        amt = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        df = df[amt.notna()].copy()
        if df.empty:
            continue
        df["amount"] = amt[amt.notna()].astype("float64")

        # Enforce sign conventions (coerce only if inconsistent)
        stake_fix = (df["Transaction_Type"] == "STAKE") & (df["amount"] > 0)
        void_fix = (df["Transaction_Type"].isin(["VOID_BET", "VOID_STAKE"])) & (df["amount"] < 0)

        if (stake_fix.any() or void_fix.any()):
            if not SIGN_WARNING_EMITTED:
                msg = (
                    "⚠️ F3: coerced transaction signs to enforce convention "
                    "(STAKE negative, VOID_* positive). Shown once."
                )
                if logger:
                    logger.warning(msg)
                elif verbose:
                    print(msg)
                SIGN_WARNING_EMITTED = True

            if stake_fix.any():
                df.loc[stake_fix, "amount"] = -df.loc[stake_fix, "amount"].abs()
            if void_fix.any():
                df.loc[void_fix, "amount"] = df.loc[void_fix, "amount"].abs()

        # Aggregate this chunk
        chunk_sum = df.groupby("Player_Profile_ID")["amount"].sum()

        # Accumulate (fast, vectorized)
        if acc.empty:
            acc = chunk_sum
        else:
            acc = acc.add(chunk_sum, fill_value=0.0)

    # Build output
    if acc.empty:
        out = pd.DataFrame(columns=["Player_Profile_ID", "f3_total_wagered"])
    else:
        out = acc.rename("f3_total_wagered").reset_index().rename(columns={"index": "Player_Profile_ID"})

    if logger:
        logger.info(f"✅ F3 klaar: {len(out):,} spelers")
        if len(out) > 0:
            logger.info(f"   Som F3: {out['f3_total_wagered'].sum():,.2f}")
            logger.info(f"   Mean F3: {out['f3_total_wagered'].mean():.2f}")

    return out


# ------------------------------
# F4: Average Wager per Day - Gemiddeld ingezet bedrag per dag (Spanish: negative)
# ------------------------------

def f4_average_wager_per_day(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F4: Average wager per active day (Spanish definition)

    Definitie:
        F4 = F3 / F1

    Waarbij:
        - F3 (participation / total wagered net of voids) verwacht <= 0 (negatief)
        - F1 (active days) >= 1 bij activiteit

    Dus F4 wordt ook verwacht <= 0 (negatief gemiddeld per actieve dag).

    Let op:
        - Geen "+1 dag" logica op datums. (Half-open interval logica zit in F3/F1.)
        - Deze functie doet dezelfde sign-consistency check als F3, maar dan op de
          F3-output (feature-waarde) i.p.v. op transactierijen:
            * Als F3 > 0, coerces naar -abs(F3) en geeft 1x warning.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f4_average_wager_per_day")
        logger.info("▶ START F4: Average Wager per Day (Spanish, negative)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # Compute F3 and F1
    f3 = f3_total_wagered(
        tables,
        x_tijdspad=x_tijdspad,
        chunksize=chunksize,
        log_path=None,
        verbose=verbose,
    )

    f1 = f1_active_days(
        tables,
        x_tijdspad=x_tijdspad,
        chunksize=chunksize,
        log_path=None,
        verbose=verbose,
    )

    if f3.empty:
        if logger:
            logger.warning("⚠️ F3 is leeg; F4 kan niet worden berekend.")
        return pd.DataFrame(columns=["Player_Profile_ID", "f4_average_wager_per_day"])

    # --- F3 sign sanity/coercion (same style as in F3: one-time warning) ---
    # Expectation: F3 <= 0. If positive due to inconsistent upstream signs, coerce.
    global SIGN_WARNING_EMITTED
    if "f3_total_wagered" in f3.columns:
        f3_bad = f3["f3_total_wagered"].notna() & (f3["f3_total_wagered"] > 0)
        if f3_bad.any():
            if not SIGN_WARNING_EMITTED:
                msg = (
                    "⚠️ F4: detected positive F3 values; coerced F3 to negative "
                    "(F3 := -abs(F3)) to enforce convention. Shown once."
                )
                if logger:
                    logger.warning(msg)
                elif verbose:
                    print(msg)
                SIGN_WARNING_EMITTED = True
            f3.loc[f3_bad, "f3_total_wagered"] = -f3.loc[f3_bad, "f3_total_wagered"].abs()

    # Merge: keep all players with F3; attach F1 if present
    merged = f3.merge(f1, on="Player_Profile_ID", how="left")

    denom = merged.get("f1_active_days")
    if denom is None:
        # F1 missing entirely
        merged["f4_average_wager_per_day"] = np.nan
    else:
        merged["f4_average_wager_per_day"] = np.where(
            denom.notna() & (denom > 0),
            merged["f3_total_wagered"] / denom,
            np.nan,
        )

    result = merged[["Player_Profile_ID", "f4_average_wager_per_day"]].copy()

    if logger:
        n_total = len(result)
        n_nan = int(result["f4_average_wager_per_day"].isna().sum())
        logger.info(f"✅ F4 klaar: {n_total:,} spelers (NaN: {n_nan:,})")
        if n_total - n_nan > 0:
            logger.info(
                f"   Mean (non-NaN): {result['f4_average_wager_per_day'].dropna().mean():.4f} "
                f"(expected <= 0)"
            )

    return result



# ------------------------------
# F5: Sex is not possible!
# ------------------------------

#
#
#

# ------------------------------
# F6: Age - Player age at end of period
# ------------------------------

def f6_age(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F6: Age (exact, birthday-correct)

    Calculate player's age in full years at the end date of x_tijdspad.
    If x_tijdspad is None, uses today's date (UTC date).

    Inputs:
        - WOK_Player_Profile: Player_Profile_DOB (date)
        - Optional: Player_Profile_Modified or Extraction_Date to pick the latest record

    Output:
        - f6_age: int (years), filtered to [18, 120] like your original code
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f6_age")
        logger.info("▶ START F6: Age")
    else:
        logger = None

    # Reference date (as DATE, not datetime)
    if x_tijdspad:
        ref_ts = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
        reference_date = pd.Timestamp(ref_ts).date()
        if logger:
            logger.info(f"  Age calculated as of (date): {x_tijdspad[1]}")
    else:
        # reproducible "today" as UTC date
        reference_date = pd.Timestamp.utcnow().date()
        if logger:
            logger.info(f"  Age calculated as of (UTC date): {reference_date.isoformat()}")

    profile_paths = tables.get("WOK_Player_Profile")
    if not profile_paths:
        if logger:
            logger.warning("⚠️ WOK_Player_Profile niet beschikbaar")
        return pd.DataFrame(columns=["Player_Profile_ID", "f6_age"])

    # We'll store best record per player: (rank_timestamp, dob_date)
    # rank_timestamp is used to keep the latest profile record if we can.
    best: Dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}

    # Sniff headers per file — only read files that actually have Player_Profile_DOB.
    # Case-tolerant: het organisatie-formaat is snake_case (player_profile_dob); iter_csv_chunks hernoemt
    # de gevraagde CamelCase-naam later terug, dus we toetsen hier case-insensitief.
    desired_cols = ["Player_Profile_DOB", "Player_Profile_Modified", "Extraction_Date"]
    paths_with_dob: list[tuple[Path, set]] = []
    for p in profile_paths:
        file_cols = set(pd.read_csv(p, nrows=0).columns.tolist())
        file_cols_lower = {c.lower() for c in file_cols}
        if "player_profile_dob" not in file_cols_lower:
            msg = f"  ⚠️ F6: {p.name} mist Player_Profile_DOB — wordt overgeslagen"
            if logger:
                logger.warning(msg)
            print(msg)
        else:
            paths_with_dob.append((p, file_cols))

    if not paths_with_dob:
        msg = "  ⚠️ Player_Profile_DOB ontbreekt in alle bestanden – f6_age wordt leeg teruggegeven"
        if logger:
            logger.warning(msg)
        print(msg)
        return pd.DataFrame(columns=["Player_Profile_ID", "f6_age"])

    for p, file_cols in paths_with_dob:
        file_cols_lower = {c.lower() for c in file_cols}
        usecols = ["Player_Profile_ID"] + [c for c in desired_cols if c.lower() in file_cols_lower]
        for df in iter_csv_chunks(
            paths=[p],
            usecols=usecols,
            chunksize=chunksize,
            verbose=verbose,
        ):
            # Keep rows with id + dob
            df = df[df["Player_Profile_ID"].notna() & df["Player_Profile_DOB"].notna()].copy()
            if df.empty:
                continue

            # Parse DOB robustly (handles ISO 'YYYY-MM-DD' én '5/5/1975').
            # organisatie-DOB is ISO; sommige pandas-versies maken ISO-datums NaT met dayfirst=True
            # (lezen het jaartal als dag), dus parse eerst expliciet ISO en val daarna pas terug
            # op dayfirst voor niet-ISO (oude) formaten.
            dob = pd.to_datetime(df["Player_Profile_DOB"], errors="coerce", format="ISO8601", utc=False)
            _na = dob.isna()
            if _na.any():
                dob = dob.mask(_na, pd.to_datetime(
                    df["Player_Profile_DOB"], errors="coerce", dayfirst=True, utc=False))
            df["dob"] = dob
            df = df[df["dob"].notna()].copy()
            if df.empty:
                continue

            # Determine a "recency" timestamp to select the latest record per player.
            # Prefer Player_Profile_Modified, else Extraction_Date, else fallback to NaT -> treated as oldest.
            recency = None
            if "Player_Profile_Modified" in df.columns:
                recency = pd.to_datetime(df["Player_Profile_Modified"], errors="coerce", utc=True)
            if recency is None or recency.isna().all():
                if "Extraction_Date" in df.columns:
                    recency = pd.to_datetime(df["Extraction_Date"], errors="coerce", utc=True)
            if recency is None:
                recency = pd.Series([pd.NaT] * len(df), index=df.index)

            df["rank_ts"] = recency.fillna(pd.Timestamp.min.tz_localize("UTC"))

            # Update best per player
            for row in df.itertuples(index=False):
                # row fields depend on df columns order; use getattr safely
                pid = getattr(row, "Player_Profile_ID")
                dob_ts = getattr(row, "dob")
                rank_ts = getattr(row, "rank_ts")

                # Keep the most recent record
                prev = best.get(pid)
                if prev is None or rank_ts > prev[0]:
                    best[pid] = (rank_ts, dob_ts)

    if not best:
        return pd.DataFrame(columns=["Player_Profile_ID", "f6_age"])

    # Compute age exactly
    records = []
    ref_y, ref_m, ref_d = reference_date.year, reference_date.month, reference_date.day

    for pid, (_, dob_ts) in best.items():
        dob_date = pd.Timestamp(dob_ts).date()
        y = ref_y - dob_date.year
        # subtract 1 if birthday hasn't occurred yet this year
        if (ref_m, ref_d) < (dob_date.month, dob_date.day):
            y -= 1

        if 18 <= y <= 120:
            records.append({"Player_Profile_ID": pid, "f6_age": int(y)})

    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=["Player_Profile_ID", "f6_age"])

    if logger:
        logger.info(f"✅ F6 Age klaar: {len(result):,} spelers")
        if len(result) > 0:
            logger.info(f"   Gemiddelde leeftijd: {result['f6_age'].mean():.1f}")
            logger.info(f"   Min: {result['f6_age'].min()}, Max: {result['f6_age'].max()}")

    return result

# ------------------------------
# F7: RTP Deviation - Deviation from expected RTP of 0.95
# ------------------------------

def f7_rtp_deviation(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F7 (Spanish): Difference in absolute value of the RTP from 0.95.

    Definition (operationalized for negative F3):
        prizes = sum(Transaction_Amount where Transaction_Type == "WINNING")
        rtp = prizes / abs(F3)
        F7 = abs(0.95 - rtp)

    Notes:
      - We do NOT apply any "+1 day" logic. Date filter is [start, end) if used.
      - We compute F3 via f3_total_wagered() and use abs(F3) as denominator
        because your F3 convention is negative.
      - One-time warning: if we must coerce transaction signs (STAKE -> negative,
        WINNING -> positive), we emit a warning once (same global flag as F3).
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f7_rtp_deviation")
        logger.info("▶ START F7: RTP Deviation (Spanish)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # Compute F3 first (player-level)
    f3 = f3_total_wagered(
        tables,
        x_tijdspad=x_tijdspad,
        chunksize=chunksize,
        log_path=None,
        verbose=verbose,
    )

    if f3.empty:
        return pd.DataFrame(columns=["Player_Profile_ID", "f7_rtp_deviation"])

    # Parse tijdspad (for filtering winnings consistently)
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f7_rtp_deviation"])

    # Accumulator for prizes (WINNING) per player
    prizes = pd.Series(dtype="float64")  # index=Player_Profile_ID, values=sum(WINNING)

    global SIGN_WARNING_EMITTED

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=[
            "Player_Profile_ID",
            "Transaction_Amount",
            "Transaction_Datetime",
            "Transaction_Type",
            "Transaction_Status",
        ],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        # Time filtering: [start, end) WITHOUT any "+1 day" logic
        if start_datum is not None:
            ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask = (ts >= start_datum) & (ts < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        # Successful only
        df = df[df["Transaction_Status"] == "SUCCESSFUL"]
        if df.empty:
            continue

        # Numeric amount
        df["amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        df = df[df["amount"].notna()].copy()
        if df.empty:
            continue

        # --- One-time sign coercion checks (same style/flag as F3) ---
        stake_fix = (df["Transaction_Type"] == "STAKE") & (df["amount"] > 0)
        win_fix = (df["Transaction_Type"] == "WINNING") & (df["amount"] < 0)

        if (stake_fix.any() or win_fix.any()):
            if not SIGN_WARNING_EMITTED:
                msg = (
                    "⚠️ F7: coerced transaction signs to enforce convention "
                    "(STAKE negative, WINNING positive). Shown once."
                )
                if logger:
                    logger.warning(msg)
                elif verbose:
                    print(msg)
                SIGN_WARNING_EMITTED = True

            if stake_fix.any():
                df.loc[stake_fix, "amount"] = -df.loc[stake_fix, "amount"].abs()
            if win_fix.any():
                df.loc[win_fix, "amount"] = df.loc[win_fix, "amount"].abs()

        # Keep WINNING only for prizes
        w = df[df["Transaction_Type"] == "WINNING"]
        if w.empty:
            continue

        chunk_sum = w.groupby("Player_Profile_ID")["amount"].sum()

        if prizes.empty:
            prizes = chunk_sum
        else:
            prizes = prizes.add(chunk_sum, fill_value=0.0)

    # Merge prizes onto F3 (keep all F3 players)
    out = f3.copy()
    out = out.merge(
        prizes.rename("prizes").reset_index().rename(columns={"index": "Player_Profile_ID"}),
        on="Player_Profile_ID",
        how="left",
    )
    out["prizes"] = out["prizes"].fillna(0.0)

    # Denominator: abs(F3) (because your F3 is negative by convention)
    denom = out["f3_total_wagered"].abs()

    # rtp = prizes / abs(F3); deviation = abs(0.95 - rtp)
    out["f7_rtp_deviation"] = np.where(
        denom > 0,
        (0.95 - (out["prizes"] / denom)).abs(),
        np.nan,
    )

    result = out[["Player_Profile_ID", "f7_rtp_deviation"]].copy()

    if logger:
        valid = result["f7_rtp_deviation"].dropna()
        logger.info(f"✅ F7 klaar: {len(result):,} spelers (NaN: {result['f7_rtp_deviation'].isna().sum():,})")
        if len(valid) > 0:
            logger.info(f"   Mean deviation (non-NaN): {valid.mean():.4f}")

    return result


# ------------------------------
# F8: Interactions per Day - Gemiddeld aantal interacties per dag
# ------------------------------

def f8_interactions_per_day(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F8 (Spanish): interactions per active day

        F8 = (#bets + #game_sessions) / F1

    Where:
      - bet interaction = bet event (we exclude BET_CANCELLED by default)
      - game interaction = a game session (counted once per player per session)
      - F1 comes from f1_active_days() to keep definitions consistent.

    Notes:
      - No '+1 day' logic. Date filter is [start, end).
      - We explicitly drop rows with missing timestamps (NaT) before counting,
        to avoid counting events that cannot be time-filtered.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f8_interactions_per_day")
        logger.info("▶ START F8: Interactions per Day (Spanish)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # Parse tijdspad
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    # Accumulator: interactions per player
    interactions = pd.Series(dtype="float64")  # index=Player_Profile_ID, values=count

    # ---- 1) Bets ----
    # organisatie-relationeel: speler via WOK_Bet_Transaction (join op pk_id); 1 interactie per bet per speler.
    bet_paths = tables.get("WOK_Bet")
    if bet_paths:
        _bet_tx_map = build_bet_tx_map(tables.get("WOK_Bet_Transaction"), chunksize=chunksize)
        for df in iter_csv_chunks(
            paths=bet_paths,
            usecols=["pk_id", "Bet_Start_Datetime", "Bet_Status"],
            chunksize=chunksize,
            verbose=verbose,
        ):
            if df.empty:
                continue

            ts = pd.to_datetime(df["Bet_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            # Drop rows with missing timestamp (cannot be filtered consistently)
            keep = ts.notna()
            if not keep.any():
                continue
            df = df.loc[keep].copy()
            ts = ts.loc[keep]

            if start_datum is not None:
                mask = (ts >= start_datum) & (ts < eind_datum)
                if not mask.any():
                    continue
                df = df.loc[mask].copy()

            # Exclude cancelled bets by default
            df = df[df["Bet_Status"].isin(["BET_PLACED", "BET_UPDATED", "BET_SETTLED"])]
            if df.empty:
                continue

            counts: Dict[str, int] = {}
            pk_col = df.columns.get_loc("pk_id")

            for i in range(len(df)):
                pk = df.iat[i, pk_col]
                players = {str(p) for p, _t in _bet_tx_map.get(pk, [])
                           if p is not None and not (isinstance(p, float) and pd.isna(p))}
                for pid in players:
                    counts[pid] = counts.get(pid, 0) + 1

            if counts:
                chunk = pd.Series(counts, dtype="float64")
                interactions = chunk if interactions.empty else interactions.add(chunk, fill_value=0.0)

    # ---- 2) Game sessions ----
    # organisatie-relationeel: speler via WOK_Game_Session_Transaction (join op pk_id); 1 interactie per sessie per speler.
    session_paths = tables.get("WOK_Game_Session")
    if session_paths:
        _session_tx_map = build_session_tx_map(tables.get("WOK_Game_Session_Transaction"), chunksize=chunksize)
        for df in iter_csv_chunks(
            paths=session_paths,
            usecols=["pk_id", "Game_Session_Start_Datetime"],
            chunksize=chunksize,
            verbose=verbose,
        ):
            if df.empty:
                continue

            ts = pd.to_datetime(df["Game_Session_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            # Drop rows with missing timestamp (cannot be filtered consistently)
            keep = ts.notna()
            if not keep.any():
                continue
            df = df.loc[keep].copy()
            ts = ts.loc[keep]

            if start_datum is not None:
                mask = (ts >= start_datum) & (ts < eind_datum)
                if not mask.any():
                    continue
                df = df.loc[mask].copy()

            # Count 1 interaction per session per player (dedupe players within same session)
            counts: Dict[str, int] = {}
            pk_col = df.columns.get_loc("pk_id")

            for i in range(len(df)):
                spk = df.iat[i, pk_col]
                players = {str(p) for p, _txid in _session_tx_map.get(spk, [])
                           if p is not None and not (isinstance(p, float) and pd.isna(p))}
                for pid in players:
                    counts[pid] = counts.get(pid, 0) + 1

            if counts:
                chunk = pd.Series(counts, dtype="float64")
                interactions = chunk if interactions.empty else interactions.add(chunk, fill_value=0.0)

    if interactions.empty:
        return pd.DataFrame(columns=["Player_Profile_ID", "f8_interactions_per_day"])

    # ---- 3) Divide by F1 (consistent definition) ----
    f1 = f1_active_days(
        tables,
        x_tijdspad=x_tijdspad,
        chunksize=chunksize,
        log_path=None,
        verbose=verbose,
    )
    if f1.empty:
        if logger:
            logger.warning("⚠️ F1 is leeg; F8 kan niet worden berekend.")
        return pd.DataFrame(columns=["Player_Profile_ID", "f8_interactions_per_day"])

    out = interactions.rename("n_interactions").reset_index().rename(columns={"index": "Player_Profile_ID"})
    out["Player_Profile_ID"] = out["Player_Profile_ID"].astype(str)
    f1["Player_Profile_ID"] = f1["Player_Profile_ID"].astype(str)
    out = out.merge(f1, on="Player_Profile_ID", how="left")

    denom = out["f1_active_days"]
    out["f8_interactions_per_day"] = np.where(
        denom.notna() & (denom > 0),
        out["n_interactions"] / denom,
        np.nan,
    )

    result = out[["Player_Profile_ID", "f8_interactions_per_day"]].copy()

    if logger:
        logger.info(f"✅ F8 klaar: {len(result):,} spelers")
        valid = result["f8_interactions_per_day"].dropna()
        if len(valid) > 0:
            logger.info(f"   Mean (non-NaN): {valid.mean():.4f}")

    return result

# ------------------------------
# F9: Big Wins per Day - wins where prize > wager, divided by F1
# ------------------------------

def f9_big_wins_per_day(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F9 (Spanish): Number of game prizes obtained during the specified period, for which
    the player receives an amount higher than the amount wagered in the game, divided by F1.

    Implementation (exact given tx-id linkage):
      - Build lookup from WOK_Player_Account_Transaction: Transaction_ID -> (type, amount)
        filtered to [start, end) by Transaction_Datetime (no +1 day logic).
      - For WOK_Game_Session:
          use Game_Transactions to collect tx-ids per session per player
          big win session if sum(WINNING) > sum(abs(STAKE)) for that player in that session
      - For WOK_Bet:
          use Bet_Transactions to collect tx-ids per bet per player
          big win bet if sum(WINNING) > sum(abs(STAKE)) for that player in that bet

    Notes:
      - Uses F1 (f1_active_days) as denominator (keeps definition consistent).
      - Sign enforcement: STAKE negative, WINNING positive; one-time warning via SIGN_WARNING_EMITTED.
      - If a bet/session has stake_sum == 0 (e.g., cancelled), it cannot be a "big win" and is ignored.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f9_big_wins_per_day")
        logger.info("▶ START F9: Big Wins per Day (WINNING > STAKE) for sessions + bets")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # Denominator: F1 active days
    f1 = f1_active_days(
        tables,
        x_tijdspad=x_tijdspad,
        chunksize=chunksize,
        log_path=None,
        verbose=verbose,
    )
    if f1.empty:
        if logger:
            logger.warning("⚠️ F1 is leeg; F9 kan niet worden berekend.")
        return pd.DataFrame(columns=["Player_Profile_ID", "f9_big_wins_per_day"])

    f1_dict = f1.set_index("Player_Profile_ID")["f1_active_days"].to_dict()

    # Parse tijdspad
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f9_big_wins_per_day"])

    # --- 1) Build tx lookup for the period: tx_id -> (type, amount) ---
    tx_type: Dict[str, str] = {}
    tx_amount: Dict[str, float] = {}

    global SIGN_WARNING_EMITTED

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=[
            "Transaction_ID",
            "Transaction_Datetime",
            "Transaction_Amount",
            "Transaction_Type",
            "Transaction_Status",
        ],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Transaction_ID"].notna()].copy()
        if df.empty:
            continue

        # Time filter on transaction datetime: [start, end)
        if start_datum is not None:
            ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            keep = ts.notna()
            if not keep.any():
                continue
            df = df.loc[keep].copy()
            ts = ts.loc[keep]
            mask = (ts >= start_datum) & (ts < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        # SUCCESSFUL only
        df = df[df["Transaction_Status"] == "SUCCESSFUL"]
        if df.empty:
            continue

        # Only STAKE/WINNING needed for comparison
        df = df[df["Transaction_Type"].isin(["STAKE", "WINNING"])]
        if df.empty:
            continue

        df["amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        df = df[df["amount"].notna()].copy()
        if df.empty:
            continue

        # One-time sign coercion
        stake_fix = (df["Transaction_Type"] == "STAKE") & (df["amount"] > 0)
        win_fix = (df["Transaction_Type"] == "WINNING") & (df["amount"] < 0)
        if (stake_fix.any() or win_fix.any()):
            if not SIGN_WARNING_EMITTED:
                msg = (
                    "⚠️ F9: coerced transaction signs to enforce convention "
                    "(STAKE negative, WINNING positive). Shown once."
                )
                if logger:
                    logger.warning(msg)
                elif verbose:
                    print(msg)
                SIGN_WARNING_EMITTED = True
            if stake_fix.any():
                df.loc[stake_fix, "amount"] = -df.loc[stake_fix, "amount"].abs()
            if win_fix.any():
                df.loc[win_fix, "amount"] = df.loc[win_fix, "amount"].abs()

        # Populate lookup (last one wins if duplicates)
        for row in df.itertuples(index=False):
            tid = row.Transaction_ID
            tx_type[tid] = row.Transaction_Type
            tx_amount[tid] = float(row.amount)

    if not tx_type:
        return pd.DataFrame(columns=["Player_Profile_ID", "f9_big_wins_per_day"])

    # --- 2) Count big-win "games" per player ---
    # We'll count big-win sessions + big-win bets (each counts as 1 if WINNING_sum > STAKE_sum)
    bigwins: Dict[str, int] = {}

    # 2a) Game sessions
    # organisatie-relationeel: tx-refs per sessie via WOK_Game_Session_Transaction (join op pk_id).
    session_paths = tables.get("WOK_Game_Session")
    if session_paths:
        _session_tx_map = build_session_tx_map(tables.get("WOK_Game_Session_Transaction"), chunksize=chunksize)
        for df in iter_csv_chunks(
            paths=session_paths,
            usecols=["pk_id", "Game_Session_Start_Datetime"],
            chunksize=chunksize,
            verbose=verbose,
        ):
            if df.empty:
                continue

            # Filter sessions by start datetime if tijdspad is provided (and drop NaT)
            ts = pd.to_datetime(df["Game_Session_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            keep = ts.notna()
            if not keep.any():
                continue
            df = df.loc[keep].copy()
            ts = ts.loc[keep]

            if start_datum is not None:
                mask = (ts >= start_datum) & (ts < eind_datum)
                if not mask.any():
                    continue
                df = df.loc[mask].copy()

            if df.empty:
                continue

            pk_col = df.columns.get_loc("pk_id")

            for i in range(len(df)):
                spk = df.iat[i, pk_col]

                stake_sum: Dict[str, float] = {}
                win_sum: Dict[str, float] = {}

                for pid_raw, tid in _session_tx_map.get(spk, []):
                    if pid_raw is None or tid is None or (isinstance(tid, float) and pd.isna(tid)):
                        continue
                    pid = str(pid_raw)
                    typ = tx_type.get(tid)
                    if typ is None:
                        continue
                    amt = tx_amount.get(tid)
                    if amt is None:
                        continue

                    if typ == "STAKE":
                        stake_sum[pid] = stake_sum.get(pid, 0.0) + abs(float(amt))
                    elif typ == "WINNING":
                        win_sum[pid] = win_sum.get(pid, 0.0) + float(amt)

                for pid, s in stake_sum.items():
                    if s <= 0:
                        continue
                    w = win_sum.get(pid, 0.0)
                    if w > s:
                        bigwins[pid] = bigwins.get(pid, 0) + 1

    # 2b) Bets (sports bets)
    # organisatie-relationeel: tx-refs per bet via WOK_Bet_Transaction (join op pk_id).
    bet_paths = tables.get("WOK_Bet")
    if bet_paths:
        _bet_tx_map = build_bet_tx_map(tables.get("WOK_Bet_Transaction"), chunksize=chunksize)
        for df in iter_csv_chunks(
            paths=bet_paths,
            usecols=["pk_id", "Bet_Start_Datetime", "Bet_Status"],
            chunksize=chunksize,
            verbose=verbose,
        ):
            if df.empty:
                continue

            # Optional: filter bet rows by bet start datetime (drop NaT)
            ts = pd.to_datetime(df["Bet_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            keep = ts.notna()
            if not keep.any():
                continue
            df = df.loc[keep].copy()
            ts = ts.loc[keep]

            if start_datum is not None:
                mask = (ts >= start_datum) & (ts < eind_datum)
                if not mask.any():
                    continue
                df = df.loc[mask].copy()

            # Exclude cancelled bets (stake_sum typically 0 anyway, but this is cleaner)
            df = df[df["Bet_Status"].isin(["BET_PLACED", "BET_UPDATED", "BET_SETTLED"])]
            if df.empty:
                continue

            pk_col = df.columns.get_loc("pk_id")

            for i in range(len(df)):
                bpk = df.iat[i, pk_col]

                stake_sum: Dict[str, float] = {}
                win_sum: Dict[str, float] = {}

                # WOK_Bet_Transaction may contain multiple tx refs (stake, winning, cashout...)
                for pid_raw, tid in _bet_tx_map.get(bpk, []):
                    if pid_raw is None or tid is None or (isinstance(tid, float) and pd.isna(tid)):
                        continue
                    pid = str(pid_raw)
                    typ = tx_type.get(tid)
                    if typ is None:
                        continue
                    amt = tx_amount.get(tid)
                    if amt is None:
                        continue

                    if typ == "STAKE":
                        stake_sum[pid] = stake_sum.get(pid, 0.0) + abs(float(amt))
                    elif typ == "WINNING":
                        win_sum[pid] = win_sum.get(pid, 0.0) + float(amt)

                for pid, s in stake_sum.items():
                    if s <= 0:
                        continue
                    w = win_sum.get(pid, 0.0)
                    if w > s:
                        bigwins[pid] = bigwins.get(pid, 0) + 1

    # --- 3) Divide by F1 ---
    all_players = set(f1_dict.keys()) | set(bigwins.keys())
    records = []
    for pid in all_players:
        bw = bigwins.get(pid, 0)
        days = f1_dict.get(pid, 0)
        per_day = (bw / days) if days and days > 0 else (0.0 if bw == 0 else np.nan)
        records.append({"Player_Profile_ID": pid, "f9_big_wins_per_day": per_day})

    if records:
        result = pd.DataFrame.from_records(records)
    else:
        print("   ⚠️  F9: Geen game session / bet data gevonden voor deze operator — lege DataFrame geretourneerd.")
        if logger:
            logger.warning("⚠️  F9: Geen game session / bet data gevonden voor deze operator — lege DataFrame geretourneerd.")
        result = pd.DataFrame(columns=["Player_Profile_ID", "f9_big_wins_per_day"])

    if logger:
        logger.info(f"✅ F9 klaar: {len(result):,} spelers")
        valid = result["f9_big_wins_per_day"].dropna()
        if len(valid) > 0:
            logger.info(f"   Mean (non-NaN): {valid.mean():.4f}")

    return result

# ------------------------------
# F10: Canceled Withdrawals per Day
# ------------------------------

def f10_canceled_withdrawals_per_day(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F10 (Spanish): Canceled withdrawals per active day

        F10 = (#WITHDRAWAL transactions with status UNSUCCESSFUL) / F1

    Notes:
      - No '+1 day' logic. Date filter is [start, end).
      - Time parsing uses ISO8601 for speed/consistency.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f10_canceled_withdrawals_per_day")
        logger.info("▶ START F10: Canceled Withdrawals per Day")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # Denominator: F1 active days
    f1 = f1_active_days(
        tables,
        x_tijdspad=x_tijdspad,
        chunksize=chunksize,
        log_path=None,
        verbose=verbose,
    )
    if f1.empty:
        if logger:
            logger.warning("⚠️ F1 is leeg; F10 kan niet worden berekend.")
        return pd.DataFrame(columns=["Player_Profile_ID", "f10_canceled_withdrawals_per_day"])

    f1_dict = f1.set_index("Player_Profile_ID")["f1_active_days"].to_dict()

    # Parse tijdspad
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f10_canceled_withdrawals_per_day"])

    canceled_per_speler: Dict[str, int] = {}

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        # Time filtering [start, end)
        if start_datum is not None:
            ts = pd.to_datetime(
                df["Transaction_Datetime"],
                errors="coerce",
                utc=True,
                format="ISO8601",
            ).dt.tz_localize(None)

            keep = ts.notna()
            if not keep.any():
                continue

            df = df.loc[keep].copy()
            ts = ts.loc[keep]

            mask = (ts >= start_datum) & (ts < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        # Filter: WITHDRAWAL + UNSUCCESSFUL
        df = df[(df["Transaction_Type"] == "WITHDRAWAL") & (df["Transaction_Status"] == "UNSUCCESSFUL")]
        if df.empty:
            continue

        counts = df.groupby("Player_Profile_ID").size()
        for player_id, count in counts.items():
            canceled_per_speler[player_id] = canceled_per_speler.get(player_id, 0) + int(count)

    # Build output: only players that exist in F1 (consistent denominator definition)
    records = []
    for player_id, active_days in f1_dict.items():
        canceled = canceled_per_speler.get(player_id, 0)
        per_day = (canceled / active_days) if active_days and active_days > 0 else np.nan
        records.append(
            {
                "Player_Profile_ID": player_id,
                "f10_canceled_withdrawals_per_day": per_day,
            }
        )

    result = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ F10 klaar: {len(result):,} spelers")
        valid = result["f10_canceled_withdrawals_per_day"].dropna()
        if len(valid) > 0:
            logger.info(f"   Mean (non-NaN): {valid.mean():.4f}")

    return result

# ------------------------------
# F11: Withdrawals per Day
# ------------------------------

def f11_withdrawals_per_day(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F11: Withdrawals per Day

    Bereken het gemiddeld aantal succesvolle withdrawals per dag.

    Output:
        - f11_withdrawals_per_day: Float >= 0
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f11_withdrawals_per_day")
        logger.info("▶ START F11: Withdrawals per Day")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    withdrawals_per_speler: Dict[str, int] = {}

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f11_withdrawals_per_day"])

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()

        # Filter: WITHDRAWAL met SUCCESSFUL status
        df = df[(df["Transaction_Type"] == "WITHDRAWAL") & (df["Transaction_Status"] == "SUCCESSFUL")]
        if df.empty:
            continue

        counts = df.groupby("Player_Profile_ID").size()
        for player_id, count in counts.items():
            withdrawals_per_speler[player_id] = withdrawals_per_speler.get(player_id, 0) + count

    # Bereken F1 voor deling
    f1_result = f1_active_days(tables, x_tijdspad=x_tijdspad, chunksize=chunksize, log_path=None, verbose=verbose)
    f1_dict = f1_result.set_index("Player_Profile_ID")["f1_active_days"].to_dict() if not f1_result.empty else {}

    records = []
    all_players = set(withdrawals_per_speler.keys()) | set(f1_dict.keys())
    for player_id in all_players:
        withdrawals = withdrawals_per_speler.get(player_id, 0)
        active_days = f1_dict.get(player_id, 0)
        per_day = withdrawals / active_days if active_days > 0 else (0.0 if withdrawals == 0 else np.nan)
        records.append({
            "Player_Profile_ID": player_id,
            "f11_withdrawals_per_day": per_day
        })

    result = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ F11 Withdrawals per Day klaar: {len(result):,} spelers")

    return result


# ------------------------------
# F12: Deposits per Day
# ------------------------------

def f12_deposits_per_day(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F12: Deposits per Day

    Bereken het gemiddeld aantal succesvolle deposits per dag.

    Output:
        - f12_deposits_per_day: Float >= 0
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f12_deposits_per_day")
        logger.info("▶ START F12: Deposits per Day")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    deposits_per_speler: Dict[str, int] = {}

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f12_deposits_per_day"])

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()

        # Filter: DEPOSIT met SUCCESSFUL status
        df = df[(df["Transaction_Type"] == "DEPOSIT") & (df["Transaction_Status"] == "SUCCESSFUL")]
        if df.empty:
            continue

        counts = df.groupby("Player_Profile_ID").size()
        for player_id, count in counts.items():
            deposits_per_speler[player_id] = deposits_per_speler.get(player_id, 0) + count

    # Bereken F1 voor deling
    f1_result = f1_active_days(tables, x_tijdspad=x_tijdspad, chunksize=chunksize, log_path=None, verbose=verbose)
    f1_dict = f1_result.set_index("Player_Profile_ID")["f1_active_days"].to_dict() if not f1_result.empty else {}

    records = []
    all_players = set(deposits_per_speler.keys()) | set(f1_dict.keys())
    for player_id in all_players:
        deposits = deposits_per_speler.get(player_id, 0)
        active_days = f1_dict.get(player_id, 0)
        per_day = deposits / active_days if active_days > 0 else (0.0 if deposits == 0 else np.nan)
        records.append({
            "Player_Profile_ID": player_id,
            "f12_deposits_per_day": per_day
        })

    result = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ F12 Deposits per Day klaar: {len(result):,} spelers")

    return result


# ------------------------------
# F13: Canceled Deposits per Day
# ------------------------------

def f13_canceled_deposits_per_day(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F13: Canceled Deposits per Day

    Count DEPOSIT transactions with UNSUCCESSFUL status, divide by active days.

    Output:
        - f13_canceled_deposits_per_day: Float
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f13_canceled_deposits_per_day")
        logger.info("▶ START F13: Canceled Deposits per Day")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]}")
    else:
        logger = None

    # Get active days
    f1_result = f1_active_days(tables, x_tijdspad=x_tijdspad, chunksize=chunksize, log_path=None, verbose=verbose)
    f1_dict = f1_result.set_index("Player_Profile_ID")["f1_active_days"].to_dict() if not f1_result.empty else {}

    # Parse tijdspad
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f13_canceled_deposits_per_day"])

    canceled_per_speler: Dict[str, int] = {}

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        # Time filtering
        if start_datum is not None:
            ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()

        # Filter: DEPOSIT + UNSUCCESSFUL
        df = df[(df["Transaction_Type"] == "DEPOSIT") & (df["Transaction_Status"] == "UNSUCCESSFUL")]

        if not df.empty:
            counts = df.groupby("Player_Profile_ID").size()
            for player_id, count in counts.items():
                canceled_per_speler[player_id] = canceled_per_speler.get(player_id, 0) + int(count)

    # Calculate per day
    records = []
    all_players = set(canceled_per_speler.keys()) | set(f1_dict.keys())

    for player_id in all_players:
        canceled = canceled_per_speler.get(player_id, 0)
        active_days = f1_dict.get(player_id, 0)
        per_day = canceled / active_days if active_days > 0 else (0.0 if canceled == 0 else np.nan)

        records.append({
            "Player_Profile_ID": player_id,
            "f13_canceled_deposits_per_day": per_day
        })

    result = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ F13 Canceled Deposits per Day klaar: {len(result):,} spelers")

    return result

# ------------------------------
# F14: Active Period Span - Dagen tussen eerste en laatste activiteit
# ------------------------------

def f14_active_period_span(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F14: Active Period Span

    Aantal dagen tussen eerste en laatste activiteit (inclusive).

    Notes:
      - Date filter is [start, end) (no +1 filtering).
      - The +1 at the end is for inclusive span definition, not time filtering.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f14_active_period_span")
        logger.info("▶ START F14: Active Period Span")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f14_active_period_span"])

    min_dates: Dict[str, pd.Timestamp] = {}
    max_dates: Dict[str, pd.Timestamp] = {}

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime"],  # add Transaction_Status if you want SUCCESSFUL-only
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        ts = pd.to_datetime(
            df["Transaction_Datetime"],
            errors="coerce",
            utc=True,
            format="ISO8601",
        ).dt.tz_localize(None)

        keep = ts.notna()
        if not keep.any():
            continue

        df = df.loc[keep].copy()
        ts = ts.loc[keep]

        if start_datum is not None:
            mask = (ts >= start_datum) & (ts < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()
            ts = ts.loc[mask]

        df["ts"] = ts

        # OPTIONAL consistency choice:
        # df = df[df["Transaction_Status"] == "SUCCESSFUL"]

        grouped = df.groupby("Player_Profile_ID")["ts"].agg(["min", "max"])

        for player_id, row in grouped.iterrows():
            mn = row["min"]
            mx = row["max"]

            if player_id not in min_dates or mn < min_dates[player_id]:
                min_dates[player_id] = mn
            if player_id not in max_dates or mx > max_dates[player_id]:
                max_dates[player_id] = mx

    records = []
    for player_id, mn in min_dates.items():
        mx = max_dates.get(player_id)
        if mx is None:
            continue
        span_days = (mx.date() - mn.date()).days + 1
        records.append({"Player_Profile_ID": player_id, "f14_active_period_span": int(span_days)})

    result = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ F14 klaar: {len(result):,} spelers")
        if len(result) > 0:
            logger.info(f"   Gemiddelde span: {result['f14_active_period_span'].mean():.1f} dagen")

    return result


# ------------------------------
# F15: Active Day Fraction - Fractie actieve dagen in span
# ------------------------------

def f15_active_day_fraction(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F15: Active Day Fraction

    Bereken de fractie van dagen in de span waarop de speler actief was.
    Dit is F1 (Active Days) gedeeld door F14 (Active Period Span).

    Output:
        - f15_active_day_fraction: Float tussen 0 en 1
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f15_active_day_fraction")
        logger.info("▶ START F15: Active Day Fraction")
    else:
        logger = None

    # Bereken F1 (active days)
    f1_result = f1_active_days(tables, x_tijdspad=x_tijdspad, chunksize=chunksize, log_path=None, verbose=verbose)

    # Bereken F14 (active period span)
    f14_result = f14_active_period_span(tables, x_tijdspad=x_tijdspad, chunksize=chunksize, log_path=None, verbose=verbose)

    if f1_result.empty or f14_result.empty:
        if logger:
            logger.warning("⚠️ F1 of F14 is leeg, kan F15 niet berekenen")
        return pd.DataFrame(columns=["Player_Profile_ID", "f15_active_day_fraction"])

    merged = f1_result.merge(f14_result, on="Player_Profile_ID", how="inner")

    # Vectorized berekening
    merged["f15_active_day_fraction"] = np.where(
        merged["f14_active_period_span"] > 0,
        merged["f1_active_days"] / merged["f14_active_period_span"],
        np.nan
    )

    result = merged[["Player_Profile_ID", "f15_active_day_fraction"]].copy()

    if logger:
        logger.info(f"✅ F15 Active Day Fraction klaar: {len(result):,} spelers")
        if len(result) > 0:
            logger.info(f"   Gemiddelde fractie: {result['f15_active_day_fraction'].mean():.3f}")

    return result



# ------------------------------
# F16: Account Age - Days since financial activation
# ------------------------------

def f16_account_age(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F16: calendar days from financial activation to the last financial transaction
    in the specified period, with a minimum of 1 day and no upper limit.

    Financial transactions are successful STAKE, DEPOSIT and WITHDRAWAL entries
    in WOK_Player_Account_Transaction. Activation uses the first such transaction
    across the available history, including transactions before the period.
    Complete history is needed to recover the actual activation date; there is
    no fallback to registration date.

    Both date boundaries in x_tijdspad are inclusive UTC calendar dates. Without
    x_tijdspad, the last financial transaction in the available history is used.
    Players with no qualifying transaction in the period are omitted.

    Output: Player_Profile_ID and f16_account_age (integer days).
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f16_account_age")
        logger.info("▶ START F16: Days since financial activation")
        if x_tijdspad:
            logger.info(f"  Period: {x_tijdspad[0]} - {x_tijdspad[1]} (inclusive dates)")
        logger.info("  Activation: first successful financial transaction in available history")
    else:
        logger = None

    start_date = end_exclusive = None
    if x_tijdspad:
        start_date = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        end_exclusive = parse_ddmmyyyy_to_timestamp(x_tijdspad[1]) + pd.Timedelta(days=1)

    transaction_paths = tables.get("WOK_Player_Account_Transaction")
    if not transaction_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f16_account_age"])

    activation_by_player: Dict[str, pd.Timestamp] = {}
    last_by_player: Dict[str, pd.Timestamp] = {}

    for df in iter_csv_chunks(
        paths=transaction_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        financial = (
            df["Player_Profile_ID"].notna()
            & df["Player_Profile_ID"].astype(str).str.strip().ne("")
            & df["Transaction_Status"].astype(str).str.strip().str.upper().eq("SUCCESSFUL")
            & df["Transaction_Type"].astype(str).str.strip().str.upper().isin(["STAKE", "DEPOSIT", "WITHDRAWAL"])
        )
        df = df.loc[financial].copy()
        if df.empty:
            continue

        df["_timestamp"] = pd.to_datetime(
            df["Transaction_Datetime"], errors="coerce", utc=True, format="ISO8601",
        ).dt.tz_localize(None)
        df = df.loc[df["_timestamp"].notna()]
        if end_exclusive is not None:
            df = df.loc[df["_timestamp"] < end_exclusive]
        if df.empty:
            continue

        # Activation is not restricted to the lower boundary of the period.
        for player_id, first in df.groupby("Player_Profile_ID")["_timestamp"].min().items():
            if player_id not in activation_by_player or first < activation_by_player[player_id]:
                activation_by_player[player_id] = first

        period_df = df if start_date is None else df.loc[df["_timestamp"] >= start_date]
        for player_id, last in period_df.groupby("Player_Profile_ID")["_timestamp"].max().items():
            if player_id not in last_by_player or last > last_by_player[player_id]:
                last_by_player[player_id] = last

    records = [
        {"Player_Profile_ID": player_id,
         "f16_account_age": max(1, (last.date() - activation_by_player[player_id].date()).days)}
        for player_id, last in last_by_player.items()
    ]
    result = pd.DataFrame.from_records(records, columns=["Player_Profile_ID", "f16_account_age"])

    if logger:
        logger.info(f"✅ F16 Account Age klaar: {len(result):,} spelers")
        if len(result) > 0:
            logger.info(f"   Gemiddelde account age: {result['f16_account_age'].mean():.0f} dagen")

    return result


# ------------------------------
# F17: Number of Bet Countries - Unique countries of betting events
# ------------------------------

# Mapping of club keywords to country codes
CLUB_TO_COUNTRY = {
    # NEDERLAND
    "Ajax": "NL", "PSV": "NL", "Feyenoord": "NL", "AZ Alkmaar": "NL", "FC Twente": "NL",
    "FC Utrecht": "NL", "Vitesse": "NL", "SC Heerenveen": "NL", "Heerenveen": "NL",
    "FC Groningen": "NL", "Groningen": "NL", "Sparta Rotterdam": "NL", "NEC Nijmegen": "NL",
    "NEC": "NL", "Go Ahead Eagles": "NL", "Fortuna Sittard": "NL", "RKC Waalwijk": "NL",
    "PEC Zwolle": "NL", "Heracles": "NL", "Willem II": "NL", "NAC Breda": "NL",
    "Excelsior Rotterdam": "NL", "Cambuur": "NL", "Volendam": "NL", "Emmen": "NL",
    "Almere City": "NL", "Roda JC": "NL", "Roda Jc Kerkrade": "NL",
    # BELGIE
    "Club Brugge": "BE", "Anderlecht": "BE", "Genk": "BE", "Standard Liege": "BE",
    "Gent": "BE", "Antwerp": "BE", "Union Saint-Gilloise": "BE", "Cercle Brugge": "BE",
    "Mechelen": "BE", "Charleroi": "BE", "Oostende": "BE", "Kortrijk": "BE",
    "Westerlo": "BE", "STVV": "BE", "Sint-Truiden": "BE", "Eupen": "BE", "Leuven": "BE",
    # DUITSLAND
    "Bayern": "DE", "Bayern Munich": "DE", "Bayern Munchen": "DE",
    "Borussia Dortmund": "DE", "Dortmund": "DE", "BVB": "DE",
    "RB Leipzig": "DE", "Leipzig": "DE", "Bayer Leverkusen": "DE", "Leverkusen": "DE",
    "Eintracht Frankfurt": "DE", "Frankfurt": "DE", "Wolfsburg": "DE",
    "Borussia Monchengladbach": "DE", "Monchengladbach": "DE", "Gladbach": "DE",
    "Union Berlin": "DE", "Hoffenheim": "DE", "SC Freiburg": "DE", "Freiburg": "DE",
    "Werder Bremen": "DE", "Bremen": "DE", "Mainz": "DE", "Augsburg": "DE",
    "Koln": "DE", "Cologne": "DE", "FC Koln": "DE", "Bochum": "DE", "Heidenheim": "DE",
    "Darmstadt": "DE", "Schalke": "DE", "Hamburg": "DE", "Hamburger SV": "DE", "HSV": "DE",
    "Hertha Berlin": "DE", "Hertha BSC": "DE", "Hannover": "DE", "Nurnberg": "DE",
    "Kaiserslautern": "DE", "Fortuna Dusseldorf": "DE", "Dusseldorf": "DE",
    "St. Pauli": "DE", "St Pauli": "DE", "Greuther Furth": "DE", "Karlsruher": "DE",
    "Paderborn": "DE", "Magdeburg": "DE", "Elversberg": "DE", "Rostock": "DE",
    "SV Sandhausen": "DE", "Sandhausen": "DE", "SSV Jahn Regensburg": "DE", "Regensburg": "DE",
    "SpVgg Unterhaching": "DE", "Unterhaching": "DE", "FC Carl Zeiss Jena": "DE", "Jena": "DE",
    "SC Freital": "DE", "TSV Alemannia Aachen": "DE", "Aachen": "DE",
    # ENGELAND
    "Manchester United": "EN", "Man United": "EN", "Man Utd": "EN",
    "Manchester City": "EN", "Man City": "EN", "Liverpool": "EN", "Arsenal": "EN",
    "Chelsea": "EN", "Tottenham": "EN", "Spurs": "EN", "Newcastle": "EN",
    "Aston Villa": "EN", "Brighton": "EN", "West Ham": "EN", "Brentford": "EN",
    "Fulham": "EN", "Crystal Palace": "EN", "Wolverhampton": "EN", "Wolves": "EN",
    "Everton": "EN", "Nottingham Forest": "EN", "Bournemouth": "EN", "Burnley": "EN",
    "Sheffield United": "EN", "Sheffield": "EN", "Luton": "EN", "Leeds": "EN",
    "Leicester": "EN", "Southampton": "EN", "Ipswich": "EN", "Norwich": "EN",
    "Watford": "EN", "West Brom": "EN", "Middlesbrough": "EN", "Coventry": "EN",
    "Sunderland": "EN", "Hull": "EN", "Stoke": "EN", "Millwall": "EN", "Blackburn": "EN",
    "Bristol City": "EN", "Swansea": "EN", "Cardiff": "EN", "Preston": "EN",
    "Queens Park Rangers": "EN", "QPR": "EN", "Birmingham": "EN", "Reading": "EN",
    "Huddersfield": "EN",
    # WALES
    "Merthyr Town": "WA", "Llanelli": "WA", "Swansea City": "WA", "Cardiff City": "WA",
    "Wrexham": "WA", "Newport County": "WA",
    # SCHOTLAND
    "Celtic": "SC", "Rangers": "SC", "Aberdeen": "SC", "Hearts": "SC", "Hibernian": "SC",
    # SPANJE
    "Real Madrid": "ES", "Barcelona": "ES", "Atletico Madrid": "ES", "Sevilla": "ES",
    "Real Sociedad": "ES", "Real Betis": "ES", "Betis": "ES", "Villarreal": "ES",
    "Athletic Bilbao": "ES", "Bilbao": "ES", "Valencia": "ES", "Osasuna": "ES",
    "Celta Vigo": "ES", "Celta": "ES", "Mallorca": "ES", "Rayo Vallecano": "ES",
    "Getafe": "ES", "Alaves": "ES", "Las Palmas": "ES", "Granada": "ES", "Cadiz": "ES",
    "Almeria": "ES", "Girona": "ES", "Elche": "ES", "Valladolid": "ES", "Espanyol": "ES",
    "Eibar": "ES", "Leganes": "ES", "Sporting Gijon": "ES", "Gijon": "ES",
    "Zaragoza": "ES", "Tenerife": "ES", "Oviedo": "ES", "Levante": "ES", "Huesca": "ES",
    "Malaga": "ES", "Albacete": "ES",
    # ITALIE
    "Juventus": "IT", "Inter Milan": "IT", "Inter": "IT", "AC Milan": "IT", "Milan": "IT",
    "Napoli": "IT", "Roma": "IT", "AS Roma": "IT", "Lazio": "IT", "Atalanta": "IT",
    "Fiorentina": "IT", "Bologna": "IT", "Torino": "IT", "Monza": "IT", "Udinese": "IT",
    "Sassuolo": "IT", "Empoli": "IT", "Cagliari": "IT", "Verona": "IT", "Hellas Verona": "IT",
    "Lecce": "IT", "Genoa": "IT", "Salernitana": "IT", "Frosinone": "IT", "Sampdoria": "IT",
    "Parma": "IT", "Como": "IT", "Venezia": "IT", "Cremonese": "IT", "Palermo": "IT",
    "Bari": "IT", "Brescia": "IT", "Pisa": "IT", "Spezia": "IT", "Reggina": "IT",
    # TURKIJE
    "Galatasaray": "TR", "Fenerbahce": "TR", "Besiktas": "TR", "Trabzonspor": "TR",
    "Basaksehir": "TR", "Antalyaspor": "TR", "Konyaspor": "TR", "Alanyaspor": "TR",
    "Kayserispor": "TR", "Sivasspor": "TR", "Kasimpasa": "TR", "Rizespor": "TR",
    "Gaziantep": "TR", "Hatayspor": "TR", "Adana Demirspor": "TR", "Ankaragucu": "TR",
    # FRANKRIJK
    "Paris Saint-Germain": "FR", "PSG": "FR", "Marseille": "FR", "Lyon": "FR",
    "Monaco": "FR", "Lille": "FR", "Nice": "FR", "Rennes": "FR", "Lens": "FR",
    "Montpellier": "FR", "Nantes": "FR", "Strasbourg": "FR", "Toulouse": "FR",
    "Brest": "FR", "Reims": "FR", "Le Havre": "FR", "Metz": "FR", "Lorient": "FR",
    # PORTUGAL
    "Benfica": "PT", "Porto": "PT", "Sporting CP": "PT", "Sporting Lisbon": "PT",
    "Braga": "PT", "Vitoria Guimaraes": "PT", "Boavista": "PT",
    # OOSTENRIJK
    "Red Bull Salzburg": "AT", "Salzburg": "AT", "Rapid Wien": "AT", "Rapid Vienna": "AT",
    "Austria Wien": "AT", "Austria Vienna": "AT", "Sturm Graz": "AT", "LASK": "AT",
    "Wolfsberger": "AT", "SK Treibach": "AT", "Treibach": "AT",
    "FSC Eggendorf Hartberg": "AT", "Hartberg": "AT",
    # ZWITSERLAND
    "Young Boys": "CH", "Basel": "CH", "Zurich": "CH", "Servette": "CH", "Lugano": "CH",
    # GRIEKENLAND
    "Olympiacos": "GR", "Panathinaikos": "GR", "AEK Athens": "GR", "PAOK": "GR",
    # RUSLAND
    "Zenit": "RU", "Spartak Moscow": "RU", "CSKA Moscow": "RU", "Lokomotiv Moscow": "RU",
    # OEKRAINE
    "Shakhtar Donetsk": "UA", "Shakhtar": "UA", "Dynamo Kyiv": "UA", "Dynamo Kiev": "UA",
    # TSJECHIE
    "Sparta Prague": "CZ", "Slavia Prague": "CZ", "Viktoria Plzen": "CZ",
    # POLEN
    "Legia Warsaw": "PL", "Lech Poznan": "PL",
    # DENEMARKEN
    "Copenhagen": "DK", "FC Copenhagen": "DK", "Brondby": "DK", "Midtjylland": "DK",
    # ZWEDEN
    "Malmo": "SE", "AIK": "SE", "Djurgarden": "SE", "Hammarby": "SE",
    # NOORWEGEN
    "Rosenborg": "NO", "Bodo/Glimt": "NO", "Molde": "NO",
    # KROATIE
    "Dinamo Zagreb": "HR", "Hajduk Split": "HR",
    # SERVIE
    "Red Star Belgrade": "RS", "Partizan Belgrade": "RS",
    # BRAZILIE
    "Flamengo": "BR", "Palmeiras": "BR", "Corinthians": "BR", "Sao Paulo": "BR",
    "Santos": "BR", "Fluminense": "BR", "Gremio": "BR", "Internacional": "BR",
    # ARGENTINIE
    "Boca Juniors": "AR", "River Plate": "AR", "Racing Club": "AR", "Independiente": "AR",
    # MEXICO
    "Club America": "MX", "Chivas": "MX", "Cruz Azul": "MX", "Tigres": "MX", "Monterrey": "MX",
    # USA/MLS
    "LA Galaxy": "US", "Inter Miami": "US", "LAFC": "US", "Atlanta United": "US",
    # JAPAN
    "Vissel Kobe": "JP", "Yokohama F. Marinos": "JP", "Urawa Reds": "JP",
    # AUSTRALIE
    "Sydney FC": "AU", "Melbourne Victory": "AU",
    # CHINA
    "Shanghai": "CN", "Guangzhou": "CN", "Beijing Guoan": "CN",
    # SAOEDI-ARABIE
    "Al-Nassr": "SA", "Al-Hilal": "SA", "Al-Ittihad": "SA", "Al-Ahli": "SA",
}


def _detect_country_from_event(part_event: str) -> set:
    """Detect country codes from a Part_Event string. Returns set of country codes."""
    if not part_event:
        return set()

    countries = set()
    event_upper = part_event.upper()

    for club, country in CLUB_TO_COUNTRY.items():
        if club.upper() in event_upper:
            countries.add(country)

    return countries


def f17_number_of_bet_countries(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F17: Number of Bet Countries

    Number of different countries in which sporting or gambling events have
    taken place on which the gambling account registers bets during the
    specified period.

    Output:
        - f17_number_of_bet_countries: Non-negative integer, NaN if no bets registered
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f17_number_of_bet_countries")
        logger.info("▶ START F17: Number of Bet Countries")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]}")
    else:
        logger = None

    # Parse tijdspad
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    bet_paths = tables.get("WOK_Bet")
    if not bet_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f17_number_of_bet_countries"])

    # Track unique countries per player
    player_countries: Dict[str, set] = {}

    # organisatie-relationeel: spelers (WOK_Bet_Transaction) + parts (WOK_Bet_Parts) per bet, join op pk_id.
    _bet_tx_map = build_bet_tx_map(tables.get("WOK_Bet_Transaction"), chunksize=chunksize)
    _bet_parts_map = build_bet_parts_map(tables.get("WOK_Bet_Parts"), chunksize=chunksize)

    for df in iter_csv_chunks(
        paths=bet_paths,
        usecols=["pk_id", "Bet_Start_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        # Time filtering
        if start_datum is not None:
            ts = pd.to_datetime(df["Bet_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()

        col_pk = df.columns.get_loc("pk_id")

        for i in range(len(df)):
            pk = df.iat[i, col_pk]

            # Get player IDs from this bet (WOK_Bet_Transaction-join)
            player_ids = [str(p) for p, _t in _bet_tx_map.get(pk, [])
                          if p is not None and not (isinstance(p, float) and pd.isna(p))]
            if not player_ids:
                continue

            # Collect countries from all parts (WOK_Bet_Parts-join; snake_case keys)
            bet_countries = set()
            for part_obj in _bet_parts_map.get(pk, []):
                event = part_obj.get("part_event", "")
                countries = _detect_country_from_event(event)
                bet_countries.update(countries)

            # Add countries to each player
            for pid in player_ids:
                if pid not in player_countries:
                    player_countries[pid] = set()
                player_countries[pid].update(bet_countries)

    # Build result
    records = []
    for player_id, countries in player_countries.items():
        records.append({
            "Player_Profile_ID": player_id,
            "f17_number_of_bet_countries": len(countries) if countries else np.nan
        })

    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=["Player_Profile_ID", "f17_number_of_bet_countries"])

    if logger:
        logger.info(f"✅ F17 Number of Bet Countries klaar: {len(result):,} spelers")
        if len(result) > 0:
            avg_countries = result["f17_number_of_bet_countries"].mean()
            logger.info(f"   Gemiddeld aantal landen: {avg_countries:.1f}")
            max_countries = result["f17_number_of_bet_countries"].max()
            logger.info(f"   Maximum aantal landen: {max_countries:.0f}")

    return result


# ------------------------------
# F18: Number of Bet Sports - Unique sports in betting events
# ------------------------------

def f18_number_of_bet_sports(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F18: Number of Bet Sports

    Number of different sports associated with the sporting or gambling
    events for which the gambling account registers bets during the
    specified period.

    Output:
        - f18_number_of_bet_sports: Non-negative integer, NaN if no bets registered
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f18_number_of_bet_sports")
        logger.info("▶ START F18: Number of Bet Sports")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]}")
    else:
        logger = None

    # Parse tijdspad
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    bet_paths = tables.get("WOK_Bet")
    if not bet_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f18_number_of_bet_sports"])

    # Track unique sports per player
    player_sports: Dict[str, set] = {}

    # organisatie-relationeel: spelers (WOK_Bet_Transaction) + parts (WOK_Bet_Parts) per bet, join op pk_id.
    _bet_tx_map = build_bet_tx_map(tables.get("WOK_Bet_Transaction"), chunksize=chunksize)
    _bet_parts_map = build_bet_parts_map(tables.get("WOK_Bet_Parts"), chunksize=chunksize)

    for df in iter_csv_chunks(
        paths=bet_paths,
        usecols=["pk_id", "Bet_Start_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        # Time filtering
        if start_datum is not None:
            ts = pd.to_datetime(df["Bet_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()

        col_pk = df.columns.get_loc("pk_id")

        for i in range(len(df)):
            pk = df.iat[i, col_pk]

            # Get player IDs from this bet (WOK_Bet_Transaction-join)
            player_ids = [str(p) for p, _t in _bet_tx_map.get(pk, [])
                          if p is not None and not (isinstance(p, float) and pd.isna(p))]
            if not player_ids:
                continue

            # Collect sports from all parts (WOK_Bet_Parts-join; snake_case keys)
            bet_sports = set()
            for part_obj in _bet_parts_map.get(pk, []):
                sport = part_obj.get("part_sport", "")
                if sport:
                    bet_sports.add(sport)

            # Add sports to each player
            for pid in player_ids:
                if pid not in player_sports:
                    player_sports[pid] = set()
                player_sports[pid].update(bet_sports)

    # Build result
    records = []
    for player_id, sports in player_sports.items():
        records.append({
            "Player_Profile_ID": player_id,
            "f18_number_of_bet_sports": len(sports) if sports else np.nan
        })

    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=["Player_Profile_ID", "f18_number_of_bet_sports"])

    if logger:
        logger.info(f"✅ F18 Number of Bet Sports klaar: {len(result):,} spelers")
        if len(result) > 0:
            avg_sports = result["f18_number_of_bet_sports"].mean()
            logger.info(f"   Gemiddeld aantal sporten: {avg_sports:.1f}")
            max_sports = result["f18_number_of_bet_sports"].max()
            logger.info(f"   Maximum aantal sporten: {max_sports:.0f}")

    return result


# ------------------------------
# F19: Proxy for Number of Competitions - Max bet parts per bet
# ------------------------------

def f19_PROXY_FOR_nr_unique_competitions_by_max_bet_parts(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F19: PROXY for Number of Competitions

    Original definition: "Number of different competitions for which the gambling
    account registers bets during the specified period."

    NOTE: This is INFEASIBLE because CDB data does not contain competition/league
    information in Bet_Parts. There is no Part_Competition or Part_League field.

    PROXY: We use the maximum number of bet parts (legs) in any single bet as a proxy.
    A combo bet with many parts likely spans multiple competitions/events.

    Output:
        - f19_max_bet_parts: Non-negative integer, NaN if no bets registered
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f19_max_bet_parts")
        logger.info("▶ START F19: Max Bet Parts (proxy for competitions)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]}")
    else:
        logger = None

    # Parse tijdspad
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    bet_paths = tables.get("WOK_Bet")
    if not bet_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f19_max_bet_parts"])

    # Track max bet parts per player
    player_max_parts: Dict[str, int] = {}

    # organisatie-relationeel: spelers (WOK_Bet_Transaction) + parts (WOK_Bet_Parts) per bet, join op pk_id.
    _bet_tx_map = build_bet_tx_map(tables.get("WOK_Bet_Transaction"), chunksize=chunksize)
    _bet_parts_map = build_bet_parts_map(tables.get("WOK_Bet_Parts"), chunksize=chunksize)

    for df in iter_csv_chunks(
        paths=bet_paths,
        usecols=["pk_id", "Bet_Start_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        # Time filtering
        if start_datum is not None:
            ts = pd.to_datetime(df["Bet_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()

        col_pk = df.columns.get_loc("pk_id")

        for i in range(len(df)):
            pk = df.iat[i, col_pk]

            # Get player IDs from this bet (WOK_Bet_Transaction-join)
            player_ids = [str(p) for p, _t in _bet_tx_map.get(pk, [])
                          if p is not None and not (isinstance(p, float) and pd.isna(p))]
            if not player_ids:
                continue

            # Count parts in this bet (WOK_Bet_Parts-join)
            num_parts = len(_bet_parts_map.get(pk, []))

            # Update max for each player
            for pid in player_ids:
                if pid not in player_max_parts:
                    player_max_parts[pid] = num_parts
                else:
                    player_max_parts[pid] = max(player_max_parts[pid], num_parts)

    # Build result
    records = []
    for player_id, max_parts in player_max_parts.items():
        records.append({
            "Player_Profile_ID": player_id,
            "f19_max_bet_parts": max_parts if max_parts > 0 else np.nan
        })

    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=["Player_Profile_ID", "f19_max_bet_parts"])

    if logger:
        logger.info(f"✅ F19 Max Bet Parts klaar: {len(result):,} spelers")
        if len(result) > 0:
            avg_parts = result["f19_max_bet_parts"].mean()
            logger.info(f"   Gemiddeld max parts: {avg_parts:.1f}")
            max_parts = result["f19_max_bet_parts"].max()
            logger.info(f"   Maximum parts: {max_parts:.0f}")

    return result


# ------------------------------
# F20: Dutch Domestic Bets Percentage
# ------------------------------

# Dutch football clubs - Eredivisie, Keuken Kampioen Divisie, and lower leagues
DUTCH_FOOTBALL_CLUBS = {
    # Eredivisie
    "Ajax", "PSV", "PSV Eindhoven", "Feyenoord", "AZ", "AZ Alkmaar",
    "FC Twente", "Twente", "FC Utrecht", "Utrecht", "Vitesse",
    "SC Heerenveen", "Heerenveen", "FC Groningen", "Groningen",
    "Sparta Rotterdam", "Sparta", "NEC", "NEC Nijmegen",
    "Go Ahead Eagles", "Fortuna Sittard", "RKC Waalwijk", "RKC",
    "PEC Zwolle", "Zwolle", "Heracles", "Heracles Almelo",
    "Willem II", "NAC Breda", "NAC", "Excelsior", "Excelsior Rotterdam",
    "SC Cambuur", "Cambuur", "FC Volendam", "Volendam",
    "FC Emmen", "Emmen", "Almere City", "Almere City FC",
    # Keuken Kampioen Divisie (Eerste Divisie)
    "Roda JC", "Roda JC Kerkrade", "Roda Jc Kerkrade",
    "De Graafschap", "FC Eindhoven", "Eindhoven",
    "MVV", "MVV Maastricht", "Maastricht",
    "FC Dordrecht", "Dordrecht", "FC Den Bosch", "Den Bosch",
    "TOP Oss", "Oss", "Telstar", "SC Telstar",
    "Jong Ajax", "Jong PSV", "Jong AZ", "Jong FC Utrecht",
    "VVV-Venlo", "VVV Venlo", "VVV", "Venlo",
    "ADO Den Haag", "ADO", "Den Haag",
    "FC Oss", "Helmond Sport", "Helmond",
    "SBV Excelsior", "Almere", "Waalwijk",
    # Tweede Divisie / Amateurs met licentie
    "FC Lisse", "Katwijk", "Quick Boys", "Noordwijk",
    "ASWH", "Kozakken Boys", "Sparta Nijkerk",
    "HHC Hardenberg", "Staphorst", "VVOG",
    "AFC", "AFC Amsterdam", "Koninklijke HFC",
    "SC Genemuiden", "ODIN 59", "Spakenburg",
    "IJsselmeervogels", "De Treffers", "Achilles 29",
    "GVVV", "SteDoCo", "Hoek", "Hercules",
    # Extra varianten
    "Feyenoord Rotterdam", "FC Ajax", "Willem 2",
}

# Football sport names in different languages
FOOTBALL_SPORT_NAMES = {"FOOTBALL", "VOETBAL", "FUSSBALL", "CALCIO", "FUTBOL", "SOCCER"}


def _is_dutch_football_event(part_event: str) -> bool:
    """Check if a Part_Event contains any Dutch football club."""
    if not part_event:
        return False
    event_upper = part_event.upper()
    for club in DUTCH_FOOTBALL_CLUBS:
        if club.upper() in event_upper:
            return True
    return False


def _is_football_sport(part_sport: str) -> bool:
    """Check if Part_Sport is football (in any language)."""
    if not part_sport:
        return False
    return part_sport.upper() in FOOTBALL_SPORT_NAMES


def f20_dutch_domestic_bets_percentage(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F20: Dutch Domestic Bets Percentage

    Original definition: "Percentage of bets in competitions taking place in the
    country of Spain or international competitions."

    ADAPTED FOR NETHERLANDS: Percentage of football bets on Dutch domestic
    competitions vs foreign/unrecognized teams.

    Calculation:
        - Filter only FOOTBALL bets (FOOTBALL, VOETBAL, FUSSBALL, CALCIO)
        - Check if Part_Event contains a recognized Dutch club
        - Ratio = Dutch football bets / Total football bets

    Output:
        - f20_dutch_domestic_bets_pct: Float 0.0-1.0, NaN if no football bets
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f20_dutch_domestic_bets_pct")
        logger.info("▶ START F20: Dutch Domestic Bets Percentage")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]}")
    else:
        logger = None

    # Parse tijdspad
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    bet_paths = tables.get("WOK_Bet")
    if not bet_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f20_dutch_domestic_bets_pct"])

    # Track per player: total football bets and Dutch football bets
    total_football_bets: Dict[str, int] = {}
    dutch_football_bets: Dict[str, int] = {}

    # organisatie-relationeel: spelers (WOK_Bet_Transaction) + parts (WOK_Bet_Parts) per bet, join op pk_id.
    _bet_tx_map = build_bet_tx_map(tables.get("WOK_Bet_Transaction"), chunksize=chunksize)
    _bet_parts_map = build_bet_parts_map(tables.get("WOK_Bet_Parts"), chunksize=chunksize)

    for df in iter_csv_chunks(
        paths=bet_paths,
        usecols=["pk_id", "Bet_Start_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        # Time filtering
        if start_datum is not None:
            ts = pd.to_datetime(df["Bet_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()

        col_pk = df.columns.get_loc("pk_id")

        for i in range(len(df)):
            pk = df.iat[i, col_pk]

            # Get player IDs from this bet (WOK_Bet_Transaction-join)
            player_ids = [str(p) for p, _t in _bet_tx_map.get(pk, [])
                          if p is not None and not (isinstance(p, float) and pd.isna(p))]
            if not player_ids:
                continue

            # Check each part for FOOTBALL (WOK_Bet_Parts-join)
            has_football = False
            is_dutch = False

            for part_obj in _bet_parts_map.get(pk, []):
                if not isinstance(part_obj, dict):
                    continue
                sport = part_obj.get("part_sport", "")
                if _is_football_sport(sport):
                    has_football = True
                    event = part_obj.get("part_event", "")
                    if _is_dutch_football_event(event):
                        is_dutch = True
                        break  # One Dutch club is enough to mark as domestic

            if has_football:
                for pid in player_ids:
                    total_football_bets[pid] = total_football_bets.get(pid, 0) + 1
                    if is_dutch:
                        dutch_football_bets[pid] = dutch_football_bets.get(pid, 0) + 1

    # Calculate percentage
    records = []
    for player_id in total_football_bets:
        total = total_football_bets[player_id]
        dutch = dutch_football_bets.get(player_id, 0)
        pct = dutch / total if total > 0 else np.nan

        records.append({
            "Player_Profile_ID": player_id,
            "f20_dutch_domestic_bets_pct": pct
        })

    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=["Player_Profile_ID", "f20_dutch_domestic_bets_pct"])

    if logger:
        logger.info(f"✅ F20 Dutch Domestic Bets Percentage klaar: {len(result):,} spelers")
        if len(result) > 0:
            avg_pct = result["f20_dutch_domestic_bets_pct"].mean()
            logger.info(f"   Gemiddeld % Nederlands: {avg_pct:.1%}")
            foreign_bettors = (result["f20_dutch_domestic_bets_pct"] < 0.5).sum()
            logger.info(f"   Spelers met >50% buitenlands: {foreign_bettors:,}")

    return result

# ------------------------------
# F21: In euro's --- not possible
# ------------------------------

#
#
#

# ------------------------------
# F22: Participation Limit Increases (with fallback)
# ------------------------------

def f22_limit_increases(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F22 (Spanish spec): Number of times the player raises their DAILY/WEEKLY/MONTHLY
    participation limits during the reference period.

    - Primary source: Limit_Participation (Participation_Amount + Participation_Time_Window + timestamp)
    - If Limit_Participation is absent/empty/not parseable: fallback proxy using other limit columns.
      Proxy logic: look for "raise events" within any of {Limit_Deposit, Limit_Login, Limit_Balance, Limit_Game_Type},
      and count an event if ANY of those limits increases at a given timestamp.
      (If a column is missing, we just ignore it and use the rest.)

    Counting rule (important):
    - If the player raises more than one limit simultaneously, it counts as ONE raise.
    - We implement this by grouping changes by exact timestamp (to the second, after tz removal).

    Date filter:
    - No '+1' logic. Period is [start, end).

    Output:
        - Player_Profile_ID
        - f22_limit_increases: int >= 0
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f22_limit_increases")
        logger.info("▶ START F22: Participation Limit Increases (with fallback proxy)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # Parse tijdspad
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    limits_paths = tables.get("WOK_Player_Limits")
    if not limits_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f22_limit_increases"])

    # Helper: normalize timestamp to naive + second resolution (for "simultaneous" grouping)
    def _norm_ts(ts):
        if ts is None:
            return None
        try:
            t = pd.Timestamp(ts)
        except Exception:
            return None
        if t.tzinfo is not None:
            t = t.tz_localize(None)
        # collapse to second to avoid microsecond noise
        return t.floor("s")

    # Helper: time filter
    def _in_period(ts):
        if ts is None:
            return False
        if start_datum is None:
            return True
        return (ts >= start_datum) and (ts < eind_datum)

    # ---- Build history per player ----
    # organisatie-relationeel: limieten staan in losse sub-tabellen (Participation/Deposit/Login/Balance),
    # gekoppeld via wok_player_limit_pk_id → WOK_Player_Limits.pk_id → player_profile_id —
    # één sub-document per limiet-soort. Participation is, net als in de root, de PRIMAIRE bron;
    # deposit/login/balance vormen de proxy-fallback per speler.
    hist_participation: Dict[str, Dict[str, List[tuple]]] = {}  # pid -> window -> [(ts, amount)]
    hist_proxy: Dict[str, List[tuple]] = {}  # pid -> [(ts, kind, value, window)]

    _events = build_limits_events_map(
        limits_paths,
        participation_paths=tables.get("WOK_Player_Limits_Participation"),
        deposit_paths=tables.get("WOK_Player_Limits_Deposit"),
        login_paths=tables.get("WOK_Player_Limits_Login"),
        balance_paths=tables.get("WOK_Player_Limits_Balance"),
        chunksize=chunksize,
    )
    for pid, evs in _events.items():
        for ts, kind, value, win in evs:
            ts = _norm_ts(ts)
            if ts is None or value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            if not _in_period(ts):
                continue
            if kind == "participation":
                # primair pad: participation zonder time window is niet bruikbaar (zoals de root)
                if win is None:
                    continue
                w = win
                if w in {"DAILY", "DAY"}:
                    w = "DAY"
                elif w in {"WEEKLY", "WEEK"}:
                    w = "WEEK"
                elif w in {"MONTHLY", "MONTH"}:
                    w = "MONTH"
                hist_participation.setdefault(str(pid), {}).setdefault(w, []).append((ts, float(value)))
            else:
                hist_proxy.setdefault(str(pid), []).append((ts, kind, value, win))

    # ---- Count increases ----
    def _count_participation_increases(win_hist: Dict[str, List[tuple]]) -> int:
        """
        win_hist: window -> [(ts, amount)]
        Count raise events; simultaneous raises across windows count as 1 if same timestamp.
        """
        # per timestamp: did any window increase at that exact ts?
        raised_at_ts: Dict[pd.Timestamp, bool] = {}

        for win, items in win_hist.items():
            if len(items) < 2:
                continue
            items.sort(key=lambda x: x[0])
            prev = None
            for ts, amount in items:
                if prev is None:
                    prev = amount
                    continue
                if amount > prev:
                    raised_at_ts[ts] = True
                prev = amount

        return int(sum(1 for _ts, flag in raised_at_ts.items() if flag))

    def _count_proxy_increases(items: List[tuple]) -> int:
        """
        Proxy: items list of (ts, kind, value, window)
        Count raise events: at a given ts, if ANY tracked numeric value increased compared to its previous value
        for the same (kind, window) stream, count that ts once.
        """
        if not items or len(items) < 2:
            return 0

        # Split streams by (kind, window)
        streams: Dict[tuple, List[tuple]] = {}
        for ts, kind, value, win in items:
            key = (kind, win)
            streams.setdefault(key, []).append((ts, value))

        raised_at_ts: Dict[pd.Timestamp, bool] = {}

        for key, seq in streams.items():
            if len(seq) < 2:
                continue
            seq.sort(key=lambda x: x[0])

            prev = None
            for ts, value in seq:
                # only numeric comparisons
                try:
                    v = float(value)
                except Exception:
                    prev = None if prev is None else prev
                    continue

                if prev is None:
                    prev = v
                    continue
                if v > prev:
                    raised_at_ts[ts] = True
                prev = v

        return int(sum(1 for _ts, flag in raised_at_ts.items() if flag))

    # All players seen anywhere
    players = set(hist_participation.keys()) | set(hist_proxy.keys())

    records = []
    for pid in players:
        # Primary if we have any participation history with at least one window entry
        if pid in hist_participation and any(len(v) > 0 for v in hist_participation[pid].values()):
            inc = _count_participation_increases(hist_participation[pid])
        else:
            # Fallback proxy
            inc = _count_proxy_increases(hist_proxy.get(pid, []))

        records.append({"Player_Profile_ID": pid, "f22_limit_increases": int(inc)})

    if not records:
        result = pd.DataFrame(columns=["Player_Profile_ID", "f22_limit_increases"])
        if logger:
            logger.warning("⚠️ F22: geen limit data gevonden — lege DataFrame geretourneerd")
        print("  ⚠️ F22: geen limit data gevonden — lege DataFrame geretourneerd")
    else:
        result = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ F22 klaar: {len(result):,} spelers")
        if len(result) > 0:
            logger.info(f"   Totaal increases: {int(result['f22_limit_increases'].sum()):,}")

    return result

# ------------------------------
# F23: Participation Limit Decreases (with fallback)
# ------------------------------

def f23_limit_decreases(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F23 (Spanish spec): Number of times the player REDUCES their DAILY/WEEKLY/MONTHLY
    participation limits during the reference period.

    - Primary source: Limit_Participation (Participation_Amount + Participation_Time_Window + timestamp)
    - If Limit_Participation is absent/empty/not parseable: fallback proxy using other limit columns.
      Proxy logic: look for "lowering events" within any of {Limit_Deposit, Limit_Login, Limit_Balance, Limit_Game_Type},
      and count an event if ANY of those limits decreases at a given timestamp.
      (If a column is missing, we just ignore it and use the rest.)

    Counting rule (important):
    - If the player lowers more than one limit simultaneously, it counts as ONE lowering.
    - If player raises one limit and lowers another simultaneously, it counts for BOTH F22 and F23.
      (Implemented naturally: we count raises and decreases separately per timestamp.)

    Date filter:
    - No '+1' logic. Period is [start, end).

    Output:
        - Player_Profile_ID
        - f23_limit_decreases: int >= 0
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f23_limit_decreases")
        logger.info("▶ START F23: Participation Limit Decreases (with fallback proxy)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # Parse tijdspad
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    limits_paths = tables.get("WOK_Player_Limits")
    if not limits_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f23_limit_decreases"])

    # Helper: normalize timestamp to naive + second resolution
    def _norm_ts(ts):
        if ts is None:
            return None
        try:
            t = pd.Timestamp(ts)
        except Exception:
            return None
        if t.tzinfo is not None:
            t = t.tz_localize(None)
        return t.floor("s")

    def _in_period(ts):
        if ts is None:
            return False
        if start_datum is None:
            return True
        return (ts >= start_datum) and (ts < eind_datum)

    # Histories
    # organisatie-relationeel: zelfde bron als F22 — losse limiet-sub-tabellen
    # (Participation/Deposit/Login/Balance) via wok_player_limit_pk_id → WOK_Player_Limits.pk_id →
    # player_profile_id. Participation is, net als in de root, de PRIMAIRE bron; deposit/login/balance
    # vormen de proxy-fallback per speler.
    hist_participation: Dict[str, Dict[str, List[tuple]]] = {}  # pid -> window -> [(ts, amount)]
    hist_proxy: Dict[str, List[tuple]] = {}  # pid -> [(ts, kind, value, window)]

    _events = build_limits_events_map(
        limits_paths,
        participation_paths=tables.get("WOK_Player_Limits_Participation"),
        deposit_paths=tables.get("WOK_Player_Limits_Deposit"),
        login_paths=tables.get("WOK_Player_Limits_Login"),
        balance_paths=tables.get("WOK_Player_Limits_Balance"),
        chunksize=chunksize,
    )
    for pid, evs in _events.items():
        for ts, kind, value, win in evs:
            ts = _norm_ts(ts)
            if ts is None or value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            if not _in_period(ts):
                continue
            if kind == "participation":
                # primair pad: participation zonder time window is niet bruikbaar (zoals de root)
                if win is None:
                    continue
                w = win
                if w in {"DAILY", "DAY"}:
                    w = "DAY"
                elif w in {"WEEKLY", "WEEK"}:
                    w = "WEEK"
                elif w in {"MONTHLY", "MONTH"}:
                    w = "MONTH"
                hist_participation.setdefault(str(pid), {}).setdefault(w, []).append((ts, float(value)))
            else:
                hist_proxy.setdefault(str(pid), []).append((ts, kind, value, win))

    # ---- Count decreases ----
    def _count_participation_decreases(win_hist: Dict[str, List[tuple]]) -> int:
        lowered_at_ts: Dict[pd.Timestamp, bool] = {}

        for win, items in win_hist.items():
            if len(items) < 2:
                continue
            items.sort(key=lambda x: x[0])
            prev = None
            for ts, amount in items:
                if prev is None:
                    prev = amount
                    continue
                if amount < prev:
                    lowered_at_ts[ts] = True
                prev = amount

        return int(sum(1 for _ts, flag in lowered_at_ts.items() if flag))

    def _count_proxy_decreases(items: List[tuple]) -> int:
        if not items or len(items) < 2:
            return 0

        streams: Dict[tuple, List[tuple]] = {}
        for ts, kind, value, win in items:
            key = (kind, win)
            streams.setdefault(key, []).append((ts, value))

        lowered_at_ts: Dict[pd.Timestamp, bool] = {}

        for key, seq in streams.items():
            if len(seq) < 2:
                continue
            seq.sort(key=lambda x: x[0])

            prev = None
            for ts, value in seq:
                try:
                    v = float(value)
                except Exception:
                    prev = None if prev is None else prev
                    continue

                if prev is None:
                    prev = v
                    continue
                if v < prev:
                    lowered_at_ts[ts] = True
                prev = v

        return int(sum(1 for _ts, flag in lowered_at_ts.items() if flag))

    players = set(hist_participation.keys()) | set(hist_proxy.keys())

    records = []
    for pid in players:
        if pid in hist_participation and any(len(v) > 0 for v in hist_participation[pid].values()):
            dec = _count_participation_decreases(hist_participation[pid])
        else:
            dec = _count_proxy_decreases(hist_proxy.get(pid, []))

        records.append({"Player_Profile_ID": pid, "f23_limit_decreases": int(dec)})

    if not records:
        result = pd.DataFrame(columns=["Player_Profile_ID", "f23_limit_decreases"])
        if logger:
            logger.warning("⚠️ F23: geen limit data gevonden — lege DataFrame geretourneerd")
        print("  ⚠️ F23: geen limit data gevonden — lege DataFrame geretourneerd")
    else:
        result = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ F23 klaar: {len(result):,} spelers")
        if len(result) > 0:
            logger.info(f"   Totaal decreases: {int(result['f23_limit_decreases'].sum()):,}")

    return result

# ------------------------------
# F24: Payment Method Variety
# ------------------------------

def f24_payment_method_variety(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F24: Payment Method Variety

    Count the number of distinct payment instruments used by each player.

    Input:
        - WOK_Player_Account_Transaction: Transaction_Deposit_Instrument

    Output:
        - f24_payment_method_variety: Integer (count of distinct payment methods)
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f24_payment_method_variety")
        logger.info("▶ START F24: Payment Method Variety")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]}")
    else:
        logger = None

    # Parse tijdspad
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f24_payment_method_variety"])

    methods_per_speler: Dict[str, set] = {}

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Deposit_Instrument", "Transaction_Type"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        # Time filtering
        if start_datum is not None:
            ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()

        # Filter: DEPOSIT or WITHDRAWAL transactions with instrument info
        df = df[df["Transaction_Type"].isin(["DEPOSIT", "WITHDRAWAL"])]
        df = df[df["Transaction_Deposit_Instrument"].notna()]

        for _, row in df.iterrows():
            player_id = row["Player_Profile_ID"]
            instrument = str(row["Transaction_Deposit_Instrument"])

            if player_id not in methods_per_speler:
                methods_per_speler[player_id] = set()
            methods_per_speler[player_id].add(instrument)

    if methods_per_speler:
        result = pd.DataFrame([
            {"Player_Profile_ID": pid, "f24_payment_method_variety": len(methods)}
            for pid, methods in methods_per_speler.items()
        ])
    else:
        result = pd.DataFrame(columns=["Player_Profile_ID", "f24_payment_method_variety"])

    if logger:
        logger.info(f"✅ F24 Payment Method Variety klaar: {len(result):,} spelers")
        if len(result) > 0:
            logger.info(f"   Gemiddeld aantal methoden: {result['f24_payment_method_variety'].mean():.2f}")

    return result

# ------------------------------
# F25: Voluntary Suspensions - Number of voluntary suspensions (lifetime UNTIL right-side cutoff to prevent target leakage)
# ------------------------------

def f25_voluntary_suspensions(
    tables: Dict[str, List[Path]],
    *,
    # Left side of x_tijdspad is NOT enforced (lifetime lookback is fine).
    # Right side IS enforced: events after x_tijdspad[1] are excluded to
    # prevent target leakage into the Y-period.
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F25 (Spanish): Number of voluntary suspensions requested by the player.

    Left side (start date) is unbounded — we look back over the full history.
    Right side (end date) is capped at x_tijdspad[1] to prevent target leakage.

    Interpretation (pragmatic, consistent with CDB/WOK tables):
      - Count voluntary suspensions from WOK_Player_Profile by counting transitions/records
        where Player_Profile_Status indicates a voluntary suspension state.
      - We dedupe consecutive identical statuses so repeated daily snapshots don't inflate counts.

    Output:
      - f25_voluntary_suspensions: non-negative integer
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f25_voluntary_suspensions")
        logger.info("▶ START F25: Voluntary Suspensions")
    else:
        logger = None

    # Determine right-side cutoff to prevent target leakage
    cutoff_ts = None
    if x_tijdspad and len(x_tijdspad) >= 2:
        cutoff_ts = pd.Timestamp(parse_ddmmyyyy_to_timestamp(x_tijdspad[1]).normalize() + pd.Timedelta(days=1))
        if logger:
            logger.info(f"  🔒 Right-side cutoff (exclusive): {cutoff_ts} (prevents target leakage)")
    else:
        if logger:
            logger.info("  ⚠️ No x_tijdspad provided — no right-side cutoff applied")

    profile_paths = tables.get("WOK_Player_Profile")
    if not profile_paths:
        if logger:
            logger.warning("⚠️ WOK_Player_Profile niet beschikbaar")
        return pd.DataFrame(columns=["Player_Profile_ID", "f25_voluntary_suspensions"])

    # Heuristic list of status values that likely represent voluntary suspensions.
    # If your data uses different codes, extend this list.
    VOLUNTARY_STATUS = {
        "SELF_EXCLUDED_TEMP",
        "SELF_EXCLUDED_INDEF"
    }

    # We count status "events" per player by tracking changes over time.
    # If Player_Profile_Modified is present, use it to order states.
    events_per_player: Dict[str, int] = {}
    last_status_per_player: Dict[str, str] = {}

    for df in iter_csv_chunks(
        paths=profile_paths,
        usecols=["Player_Profile_ID", "Player_Profile_Status", "Player_Profile_Modified"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna() & df["Player_Profile_Status"].notna()].copy()
        if df.empty:
            continue

        # Parse modified timestamp if available, else keep original chunk order
        ts = pd.to_datetime(df["Player_Profile_Modified"], errors="coerce", utc=True).dt.tz_localize(None)
        df["__ts"] = ts

        # Filter out rows beyond the right-side cutoff to prevent target leakage
        if cutoff_ts is not None:
            df = df[df["__ts"].isna() | (df["__ts"] < cutoff_ts)]
            if df.empty:
                continue

        # Sort within chunk by player then timestamp (NaT at end)
        df["__pid"] = df["Player_Profile_ID"].astype(str)
        df["__status"] = df["Player_Profile_Status"].astype(str)

        df = df.sort_values(["__pid", "__ts"], na_position="last")

        # Walk rows (chunk-sized; acceptable) tracking per-player status transitions
        pid_col = df.columns.get_loc("__pid")
        st_col = df.columns.get_loc("__status")

        for i in range(len(df)):
            pid = df.iat[i, pid_col]
            status = df.iat[i, st_col]

            prev = last_status_per_player.get(pid)
            if prev == status:
                continue  # dedupe consecutive identical snapshots

            # status changed (or first seen)
            last_status_per_player[pid] = status

            if status in VOLUNTARY_STATUS:
                events_per_player[pid] = events_per_player.get(pid, 0) + 1

    if not events_per_player:
        result = pd.DataFrame(columns=["Player_Profile_ID", "f25_voluntary_suspensions"])
    else:
        result = pd.DataFrame(
            [{"Player_Profile_ID": pid, "f25_voluntary_suspensions": int(n)} for pid, n in events_per_player.items()]
        )

    if logger:
        logger.info(f"✅ F25 klaar: {len(result):,} spelers met ≥1 voluntary suspension event")
        if len(result) > 0:
            logger.info(f"   Totaal events: {int(result['f25_voluntary_suspensions'].sum()):,}")

    return result


# ------------------------------
# F26: balance drop frequency (per active day) - percentage of active days with balance drops below threshold (e.g. EUR 2)
# ------------------------------

def f26_balance_drop_frequency(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
    start_balances: Dict[str, float] | None = None,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """
    F26: Balance Drop Frequency (per active day)

    Spec-approx (best possible with available data):
      - Reconstruct opening balance from a snapshot, then apply account transactions.
      - Leave the feature unknown when no reliable opening balance is available.
      - Count transitions where balance crosses from >= threshold to < threshold
        **caused by a negative interaction** (delta < 0).
      - Divide by F1 (active days) for the same reference period.

    Notes:
      - No '+1 day' logic. Period filter is [start, end).
      - Enforces STAKE amounts as negative (one-time warning if coercion applied).
      - Accepts snapshots at the start and reverses later snapshots when needed.
      - Does not prefilter by EOD balance: intraday drops may recover before EOD.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f26_balance_drop_frequency")
        logger.info("▶ START F26: Balance Drop Frequency (below EUR 2 per active day)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # --- tijdspad ---
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f26_balance_drop_frequency"])

    # Shared opening balances; no EOD prefilter (intraday dips can recover by EOD).
    if start_balances is None:
        start_balances = reconstruct_start_balances(
            tables, x_tijdspad=x_tijdspad, chunksize=chunksize, verbose=verbose,
        )
    if logger:
        logger.info(f"  Reliable opening balances: {len(start_balances):,} players")

    # ------------------------------------------------------------------
    # B) Build F1 denominator (active days) for the same period
    # ------------------------------------------------------------------
    f1 = f1_active_days(
        tables,
        x_tijdspad=x_tijdspad,
        chunksize=chunksize,
        log_path=None,
        verbose=verbose,
    )
    if f1.empty:
        if logger:
            logger.warning("⚠️ F1 is leeg; F26 kan niet worden berekend.")
        return pd.DataFrame(columns=["Player_Profile_ID", "f26_balance_drop_frequency"])

    f1_dict = {str(pid): days for pid, days in f1.set_index("Player_Profile_ID")["f1_active_days"].items()}

    if not start_balances:
        return pd.DataFrame({"Player_Profile_ID": list(f1_dict), "f26_balance_drop_frequency": np.nan})

    # ------------------------------------------------------------------
    # C) Collect & compute for players with known opening balances
    # ------------------------------------------------------------------
    # Keep the existing per-player chronological sort for event detection.
    tx_buffer: Dict[str, List[Tuple[pd.Timestamp, float, str]]] = {}  # pid -> [(ts, amount, type), ...]

    stake_coercion_warned = False

    need_cols = ["Player_Profile_ID", "Transaction_Datetime", "Transaction_Amount", "Transaction_Type", "Transaction_Status"]
    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=need_cols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        df = df[df["Player_Profile_ID"].astype(str).isin(start_balances)]
        if df.empty:
            continue

        # only successful tx (you can broaden later if spec says otherwise)
        if "Transaction_Status" in df.columns:
            df = df[df["Transaction_Status"] == "SUCCESSFUL"]
            if df.empty:
                continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        amt = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        df["amt"] = amt
        df = df[np.isfinite(df["amt"])].copy()
        if df.empty:
            continue

        # enforce STAKE negative (one-time warning)
        is_stake = df["Transaction_Type"] == "STAKE"
        if is_stake.any():
            pos_stake = is_stake & (df["amt"] > 0)
            if pos_stake.any():
                df.loc[pos_stake, "amt"] = -df.loc[pos_stake, "amt"].abs()
                if (not stake_coercion_warned) and logger:
                    logger.warning("⚠️ Coerced positive STAKE amounts to negative (one-time warning).")
                stake_coercion_warned = True

        # buffer
        for row in df.itertuples(index=False):
            pid = str(getattr(row, "Player_Profile_ID"))
            t = getattr(row, "ts")
            a = float(getattr(row, "amt"))
            typ = str(getattr(row, "Transaction_Type"))
            tx_buffer.setdefault(pid, []).append((t, a, typ))

    # compute drops
    drop_counts: Dict[str, int] = {}
    for pid, txs in tx_buffer.items():
        if not txs:
            drop_counts[pid] = 0
            continue

        # sort by timestamp (stable)
        txs.sort(key=lambda x: x[0])

        bal = start_balances[pid]

        cnt = 0
        for t, delta, typ in txs:
            before = bal
            after = before + delta

            # "pak alle negatieve interactions mee"
            if delta < 0 and before >= threshold and after < threshold:
                cnt += 1

            bal = after

        drop_counts[pid] = cnt

    # ------------------------------------------------------------------
    # D) Divide by F1
    # ------------------------------------------------------------------
    records = []
    # Preserve active players, including those with unknown opening balances.
    all_players = set(f1_dict.keys()) | set(drop_counts.keys())

    for pid in all_players:
        drops = drop_counts.get(pid, 0)
        active_days = f1_dict.get(pid, 0)
        if pid not in start_balances:
            per_day = np.nan
        else:
            per_day = drops / active_days if active_days > 0 else (0.0 if drops == 0 else np.nan)
        records.append({"Player_Profile_ID": pid, "f26_balance_drop_frequency": per_day})

    result = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ F26 klaar: {len(result):,} spelers")
        valid = result["f26_balance_drop_frequency"].dropna()
        if len(valid) > 0:
            logger.info(f"   Mean (non-NaN): {valid.mean():.4f}")

    return result



# ------------------------------
# F27: Deposits after balance fell below EUR 2 (per active day)
# ------------------------------

def f27_deposits_after_balance_below_2_per_day(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
    start_balances: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    F27:
    Number of times that, after the balance has fallen below EUR 2,
    the player makes a deposit, divided by F1.

    Output:
      - f27_deposits_after_below2_per_day: float (>=0) or NaN if never below 2.

    Notes:
      - No '+1 day' logic. Filter is [start, end).
      - Uses a shared, snapshot-based opening balance; unknown stays NaN.
      - Includes ALL negative interactions (any tx with amount < 0).
      - Requires chronological processing per player. Assumes input is roughly chronological
        per file; if not, you need a pre-sort step.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f27_deposits_after_balance_below_2_per_day")
        logger.info("▶ START F27: Deposits after balance < 2 (per active day)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # ---- tijdspad ----
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    # ---- F1 (denominator) ----
    f1 = f1_active_days(tables, x_tijdspad=x_tijdspad, chunksize=chunksize, log_path=None, verbose=verbose)
    if f1.empty:
        return pd.DataFrame(columns=["Player_Profile_ID", "f27_deposits_after_below2_per_day"])

    f1_dict = {str(pid): days for pid, days in f1.set_index("Player_Profile_ID")["f1_active_days"].items()}

    if start_balances is None:
        start_balances = reconstruct_start_balances(
            tables, x_tijdspad=x_tijdspad, chunksize=chunksize, verbose=verbose,
        )
    if logger:
        logger.info(f"  Reliable opening balances: {len(start_balances):,} players")

    if not start_balances:
        return pd.DataFrame({"Player_Profile_ID": list(f1_dict), "f27_deposits_after_below2_per_day": np.nan})

    # ---- numerator counting via running balance ----
    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f27_deposits_after_below2_per_day"])

    # Running state
    running_balance: Dict[str, float] = {}        # pid -> current balance
    ever_below2: Dict[str, bool] = {}             # pid -> ever crossed/been <2 after a debit
    deposits_after_below2: Dict[str, int] = {}    # pid -> count

    threshold = 2.0

    # one-time warning on STAKE coercion (optional – you said you want this once)
    coerced_stake_warned = False

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=[
            "Player_Profile_ID",
            "Transaction_Datetime",
            "Transaction_Amount",
            "Transaction_Type",
            "Transaction_Status",
        ],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        df = df[df["Player_Profile_ID"].astype(str).isin(start_balances)]
        if df.empty:
            continue

        # Successful only (otherwise balance effects are ambiguous)
        df = df[df["Transaction_Status"] == "SUCCESSFUL"]
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()]
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        df["amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        df = df[np.isfinite(df["amount"])]
        if df.empty:
            continue

        # Enforce STAKE as negative (if not already) — per your project convention
        is_stake = df["Transaction_Type"].astype(str).eq("STAKE")
        # identify stake that is positive -> flip
        to_flip = is_stake & (df["amount"] > 0)
        if to_flip.any():
            df.loc[to_flip, "amount"] = -df.loc[to_flip, "amount"]
            if (not coerced_stake_warned) and logger:
                logger.warning("⚠️ Coerced positive STAKE amounts to negative (one-time warning).")
            coerced_stake_warned = True

        # Sort inside chunk by (pid, ts) to reduce damage
        df = df.sort_values(["Player_Profile_ID", "ts"])

        # Process row-wise (stateful)
        pid_col = df.columns.get_loc("Player_Profile_ID")
        ts_col  = df.columns.get_loc("ts")
        amt_col = df.columns.get_loc("amount")
        typ_col = df.columns.get_loc("Transaction_Type")

        for i in range(len(df)):
            pid = str(df.iat[i, pid_col])
            amt = float(df.iat[i, amt_col])
            typ = str(df.iat[i, typ_col])

            # init running balance for this pid
            if pid not in running_balance:
                running_balance[pid] = start_balances[pid]
                ever_below2.setdefault(pid, False)
                deposits_after_below2.setdefault(pid, 0)

            bal_before = running_balance[pid]

            # Apply transaction to balance
            bal_after = bal_before + amt
            running_balance[pid] = bal_after

            # If this is a debit (any negative interaction), and it pushed/kept us below 2, mark ever_below2
            if amt < 0:
                # spec-style: we care about "after a negative interaction"
                if (bal_before >= threshold) and (bal_after < threshold):
                    ever_below2[pid] = True
                # also: if already <2, we keep it True once it ever happened
                # (no action needed)

            # Count deposits that happen while balance <2 (before deposit), but only if ever_below2 True
            if typ == "DEPOSIT" and ever_below2.get(pid, False):
                if bal_before < threshold:
                    deposits_after_below2[pid] += 1

    # ---- build output anchored on F1 players ----
    records = []
    for pid, active_days in f1_dict.items():
        below = ever_below2.get(pid, False)
        if not below:
            per_day = np.nan  # N/A if never below 2
        else:
            n = deposits_after_below2.get(pid, 0)
            per_day = (n / active_days) if (active_days and active_days > 0) else np.nan
        records.append({"Player_Profile_ID": pid, "f27_deposits_after_below2_per_day": per_day})

    out = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ F27 klaar: {len(out):,} spelers (anchored on F1)")
        na = out["f27_deposits_after_below2_per_day"].isna().sum()
        logger.info(f"   N/A (unknown opening balance or never below 2): {na:,}")

    return out


# ------------------------------
# F28: Median time (seconds) from balance < 2 to deposit
# ------------------------------

def f28_median_seconds_below2_to_deposit(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
    start_balances: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    F28:
    Median time (seconds) between the player's balance falling below EUR 2
    (triggered by a negative interaction) and then making a deposit.

    Output:
      - f28_median_seconds_below2_to_deposit: float (seconds) or NaN if:
          * balance never fell below 2, OR
          * no deposit occurred after having balance < 2.

    Assumptions:
      - Uses a shared, snapshot-based opening balance; unknown stays NaN.
      - Input transactions are globally sorted by Transaction_Datetime (across files),
        so per-player state is correct without a per-PID sort step.
      - No '+1 day' logic. Filter is [start, end).
      - Counts only SUCCESSFUL transactions.
      - Negative interactions are any tx with amount < 0 (after STAKE coercion).
      - Deposit counted when Transaction_Type == "DEPOSIT" and bal_before < 2.

    Implementation notes:
      - We record episode start timestamp when a debit causes a drop from >=2 to <2.
      - For that episode, we take the FIRST subsequent deposit while bal_before < 2,
        then close the episode.
      - We can have multiple episodes per player; we take the median delta seconds.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f28_median_seconds_below2_to_deposit")
        logger.info("▶ START F28: Median seconds from balance<2 to deposit")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # ---- tijdspad ----
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    # ---- anchor output on F1 players (consistent with your other per-day features) ----
    f1 = f1_active_days(tables, x_tijdspad=x_tijdspad, chunksize=chunksize, log_path=None, verbose=verbose)
    if f1.empty:
        return pd.DataFrame(columns=["Player_Profile_ID", "f28_median_seconds_below2_to_deposit"])
    f1_players = set(f1["Player_Profile_ID"].astype(str).tolist())

    if start_balances is None:
        start_balances = reconstruct_start_balances(
            tables, x_tijdspad=x_tijdspad, chunksize=chunksize, verbose=verbose,
        )
    if logger:
        logger.info(f"  Reliable opening balances: {len(start_balances):,} players")

    if not start_balances:
        return pd.DataFrame({"Player_Profile_ID": list(f1_players), "f28_median_seconds_below2_to_deposit": np.nan})

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f28_median_seconds_below2_to_deposit"])

    threshold = 2.0

    # per-player running state
    running_balance: Dict[str, float] = {}
    # if player is currently in a "below2 episode waiting for a deposit", store start timestamp
    below2_start_ts: Dict[str, pd.Timestamp] = {}
    # store deltas per player (seconds)
    deltas_sec: Dict[str, List[float]] = {}
    # track whether balance EVER fell below 2 (for spec N/A reasons)
    ever_below2: Dict[str, bool] = {}

    coerced_stake_warned = False

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=[
            "Player_Profile_ID",
            "Transaction_Datetime",
            "Transaction_Amount",
            "Transaction_Type",
            "Transaction_Status",
        ],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        df = df[df["Player_Profile_ID"].astype(str).isin(start_balances)]
        if df.empty:
            continue

        # Successful only
        df = df[df["Transaction_Status"] == "SUCCESSFUL"]
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()]
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        df["amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        df = df[np.isfinite(df["amount"])]
        if df.empty:
            continue

        # Enforce STAKE negative (one-time warning)
        is_stake = df["Transaction_Type"].astype(str).eq("STAKE")
        to_flip = is_stake & (df["amount"] > 0)
        if to_flip.any():
            df.loc[to_flip, "amount"] = -df.loc[to_flip, "amount"]
            if (not coerced_stake_warned) and logger:
                logger.warning("⚠️ Coerced positive STAKE amounts to negative (one-time warning).")
            coerced_stake_warned = True

        # If you truly guarantee global sorting by Transaction_Datetime, you do NOT need this.
        # Keeping it would be harmful if it reorders across players inside a chunk.
        # So: do NOT sort here.

        pid_col = df.columns.get_loc("Player_Profile_ID")
        ts_col  = df.columns.get_loc("ts")
        amt_col = df.columns.get_loc("amount")
        typ_col = df.columns.get_loc("Transaction_Type")

        for i in range(len(df)):
            pid = str(df.iat[i, pid_col])
            t   = df.iat[i, ts_col]
            amt = float(df.iat[i, amt_col])
            typ = str(df.iat[i, typ_col])

            if pid not in running_balance:
                running_balance[pid] = start_balances[pid]
                ever_below2.setdefault(pid, False)
                deltas_sec.setdefault(pid, [])

            bal_before = running_balance[pid]
            bal_after  = bal_before + amt
            running_balance[pid] = bal_after

            # episode start condition: negative interaction causes drop from >=2 to <2
            if amt < 0 and (bal_before >= threshold) and (bal_after < threshold):
                ever_below2[pid] = True
                # start a new episode (overwrite any existing pending one)
                below2_start_ts[pid] = t

            # if we're in an episode, count FIRST deposit that occurs while balance is still <2 before deposit
            if typ == "DEPOSIT" and pid in below2_start_ts:
                if bal_before < threshold:
                    dt = (t - below2_start_ts[pid]).total_seconds()
                    if dt >= 0:
                        deltas_sec[pid].append(float(dt))
                    # close episode (only first deposit after drop)
                    del below2_start_ts[pid]

    # ---- compute median per player (anchored on F1 players) ----
    records = []
    for pid in f1_players:
        if not ever_below2.get(pid, False):
            med = np.nan
        else:
            vals = deltas_sec.get(pid, [])
            med = float(np.median(vals)) if vals else np.nan
        records.append({"Player_Profile_ID": pid, "f28_median_seconds_below2_to_deposit": med})

    out = pd.DataFrame.from_records(records)

    if logger:
        na = out["f28_median_seconds_below2_to_deposit"].isna().sum()
        logger.info(f"✅ F28 klaar: {len(out):,} spelers (anchored on F1)")
        logger.info(f"   N/A: {na:,}")

    return out

# ------------------------------
# F29: Sessions per active day (other games + pre-drawn) — via WOK_Game_Session
# ------------------------------

def f29_sessions_other_predrawn_per_day(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F29:
    Number of sessions (other games + pre-drawn games) during the reference period,
    divided by F1 (active days).

    Practical CDB mapping:
      - "sessions of other games" -> WOK_Game_Session rows
      - "pre-drawn games" is not clearly separable in CDB for now; we treat
        WOK_Game_Session as the session universe for this feature.

    Output:
      - f29_sessions_per_day: float >= 0, or NaN if no session has been opened (per player)

    Notes:
      - No '+1 day' logic. Filter is [start, end).
      - Anchored on F1 players (same pattern as your other per-day features).
      - Dedupe by Game_Session_ID to avoid double counting across repeated extracts.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f29_sessions_other_predrawn_per_day")
        logger.info("▶ START F29: Sessions (other+pre-drawn) per active day")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # ---- tijdspad ----
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    # ---- F1 (denominator) ----
    f1 = f1_active_days(tables, x_tijdspad=x_tijdspad, chunksize=chunksize, log_path=None, verbose=verbose)
    if f1.empty:
        return pd.DataFrame(columns=["Player_Profile_ID", "f29_sessions_per_day"])
    f1_dict = f1.set_index("Player_Profile_ID")["f1_active_days"].to_dict()

    # ---- input ----
    sess_paths = tables.get("WOK_Game_Session")
    if not sess_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f29_sessions_per_day"])

    # organisatie-relationeel: transacties per sessie uit WOK_Game_Session_Transaction (join op pk_id).
    session_tx = build_session_tx_map(tables.get("WOK_Game_Session_Transaction"), chunksize=chunksize)

    sessions_per_player: Dict[str, int] = {}
    seen_session_ids: set[str] = set()  # global dedupe on Game_Session_ID

    need_cols = ["pk_id", "Game_Session_ID", "Game_Session_Start_Datetime"]

    for df in iter_csv_chunks(
        paths=sess_paths,
        usecols=need_cols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        df = df[df["Game_Session_ID"].notna()].copy()
        if df.empty:
            continue

        # parse start time
        ts = pd.to_datetime(df["Game_Session_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()]
        if df.empty:
            continue

        # time filtering
        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()
            if df.empty:
                continue

        # sort inside chunk (not required if you've globally sorted, but cheap + stable)
        df = df.sort_values(["ts", "Game_Session_ID"], kind="mergesort")

        sid_idx = df.columns.get_loc("Game_Session_ID")
        pk_idx  = df.columns.get_loc("pk_id")
        for i in range(len(df)):
            sid = str(df.iat[i, sid_idx])

            # global dedupe: count each session once
            if sid in seen_session_ids:
                continue
            seen_session_ids.add(sid)

            pk = df.iat[i, pk_idx]
            # extract unique players within this session (relationele join i.p.v. JSON)
            pids = set()
            for pid, _txid in session_tx.get(pk, []):
                if pid is not None and str(pid).strip() != "" and str(pid).lower() != "nan":
                    pids.add(str(pid))

            # If no pid found, skip (can't attribute session)
            if not pids:
                continue

            for pid in pids:
                sessions_per_player[pid] = sessions_per_player.get(pid, 0) + 1

    # ---- build output anchored on F1 players ----
    records = []
    for pid, active_days in f1_dict.items():
        n_sessions = sessions_per_player.get(pid, 0)
        if n_sessions == 0:
            per_day = np.nan  # N/A if no session opened (per spec)
        else:
            per_day = (n_sessions / active_days) if (active_days and active_days > 0) else np.nan

        records.append({
            "Player_Profile_ID": pid,
            "f29_sessions_per_day": per_day
        })

    if records:
        out = pd.DataFrame.from_records(records)
    else:
        print("   ⚠️  F29: Geen session data gevonden voor deze operator — lege DataFrame geretourneerd.")
        if logger:
            logger.warning("⚠️  F29: Geen session data gevonden voor deze operator — lege DataFrame geretourneerd.")
        out = pd.DataFrame(columns=["Player_Profile_ID", "f29_sessions_per_day"])

    if logger:
        logger.info(f"✅ F29 klaar: {len(out):,} spelers (anchored on F1)")
        logger.info(f"   Unieke sessies gezien: {len(seen_session_ids):,}")
        logger.info(f"   N/A (no sessions): {out['f29_sessions_per_day'].isna().sum():,}")

    return out

# ------------------------------
# F30: Avg interactions per session (other games / pre-drawn) — via WOK_Game_Session.Game_Transactions
# ------------------------------

def f30_avg_interactions_per_session(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F30:
    Average number of interactions performed during a session of other games / pre-drawn games.
    Implemented as: count Transaction_IDs in Game_Transactions per session, averaged per player.

    Output:
      - f30_avg_interactions_per_session: float >= 0, or NaN if player has no sessions in period.

    Notes:
      - No '+1 day' logic. Filter is [start, end).
      - Uses reading_difficult_json.iter_transaction_ids_from_Game_Transactions
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f30_avg_interactions_per_session")
        logger.info("▶ START F30: Avg interactions per session (Game_Transactions count)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # ---- tijdspad ----
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    session_paths = tables.get("WOK_Game_Session")
    if not session_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f30_avg_interactions_per_session"])

    # organisatie-relationeel: transacties per sessie uit WOK_Game_Session_Transaction (join op pk_id).
    session_tx = build_session_tx_map(tables.get("WOK_Game_Session_Transaction"), chunksize=chunksize)

    # per player totals
    tx_total: Dict[str, int] = {}
    session_total: Dict[str, int] = {}

    # we count per session_id to avoid double counting if the same session appears multiple times
    # (extractions / replaced records). Keep only in-memory set of seen session ids.
    seen_sessions: set[str] = set()

    for df in iter_csv_chunks(
        paths=session_paths,
        usecols=["pk_id", "Game_Session_ID", "Game_Session_Start_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        # drop rows zonder session id (de transacties komen uit de relationele tabel via pk_id)
        df = df[df["Game_Session_ID"].notna()].copy()
        if df.empty:
            continue

        # tijdsfilter op start datetime
        ts = pd.to_datetime(df["Game_Session_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()]
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        # process row-wise (json parsing)
        sid_idx = df.columns.get_loc("Game_Session_ID")
        pk_idx  = df.columns.get_loc("pk_id")

        for i in range(len(df)):
            sid = str(df.iat[i, sid_idx])
            if sid in seen_sessions:
                continue
            seen_sessions.add(sid)

            pk = df.iat[i, pk_idx]

            # count txids per pid binnen deze sessie (relationele join i.p.v. JSON)
            per_pid_count: Dict[str, int] = {}
            for pid, txid in session_tx.get(pk, []):
                if pid is None or (isinstance(pid, float) and pd.isna(pid)):
                    continue
                pid = str(pid)
                per_pid_count[pid] = per_pid_count.get(pid, 0) + 1

            # if no parsable tx => ignore this session row (no player attribution possible)
            if not per_pid_count:
                continue

            for pid, n_tx in per_pid_count.items():
                tx_total[pid] = tx_total.get(pid, 0) + int(n_tx)
                session_total[pid] = session_total.get(pid, 0) + 1

    # build result
    players = sorted(set(tx_total.keys()) | set(session_total.keys()))
    records = []
    for pid in players:
        n_sessions = session_total.get(pid, 0)
        if n_sessions <= 0:
            avg = np.nan
        else:
            avg = tx_total.get(pid, 0) / n_sessions
        records.append({"Player_Profile_ID": pid, "f30_avg_interactions_per_session": avg})

    if records:
        out = pd.DataFrame.from_records(records)
    else:
        print("   ⚠️  F30: Geen session data gevonden voor deze operator — lege DataFrame geretourneerd.")
        if logger:
            logger.warning("⚠️  F30: Geen session data gevonden voor deze operator — lege DataFrame geretourneerd.")
        out = pd.DataFrame(columns=["Player_Profile_ID", "f30_avg_interactions_per_session"])

    if logger:
        logger.info(f"✅ F30 klaar: {len(out):,} spelers")
        if len(out) > 0:
            na = out["f30_avg_interactions_per_session"].isna().sum()
            logger.info(f"   N/A (no sessions): {na:,}")

    return out


def f31_median_rounds_per_session(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F31: Median Rounds per Session

    Calculate the median of Game_Session_Rounds for each player. The relational
    WOK_Game_Session_Transaction table supplies player links only; multiple
    transactions in a session do not multiply the session's round count.

    Only positive, finite integer round counts are used. Known players with no
    valid round counts receive NaN; transaction counts are not a fallback.

    Input:
        - WOK_Game_Session: pk_id, Game_Session_Start_Datetime, Game_Session_Rounds
        - WOK_Game_Session_Transaction: session-to-player links

    Output:
        - f31_median_rounds_per_session: Float (median rounds per session)
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f31_median_rounds_per_session")
        logger.info("▶ START F31: Median Rounds per Session")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]}")
    else:
        logger = None

    # Parse tijdspad
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    session_paths = tables.get("WOK_Game_Session")
    if not session_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f31_median_rounds_per_session"])

    # organisatie-relationeel: transacties per sessie uit WOK_Game_Session_Transaction (join op pk_id).
    session_tx = build_session_tx_map(tables.get("WOK_Game_Session_Transaction"), chunksize=chunksize)

    # Track rounds per session for each player
    rounds_per_session_per_speler: Dict[str, List[int]] = {}

    for df in iter_csv_chunks(
        paths=session_paths,
        usecols=["pk_id", "Game_Session_Start_Datetime", "Game_Session_Rounds"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        if "Game_Session_Rounds" not in df.columns:
            raise ValueError("F31 requires Game_Session_Rounds in WOK_Game_Session.")

        # Time filtering
        if start_datum is not None:
            ts = pd.to_datetime(df["Game_Session_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask_periode = (ts >= start_datum) & (ts < eind_datum)
            if not mask_periode.any():
                continue
            df = df.loc[mask_periode].copy()

        rounds = pd.to_numeric(df["Game_Session_Rounds"], errors="coerce")
        valid_rounds = rounds.notna() & np.isfinite(rounds) & (rounds > 0) & (rounds % 1 == 0)
        df["_rounds"] = rounds.where(valid_rounds)

        for pk, round_count in df[["pk_id", "_rounds"]].itertuples(index=False, name=None):
            players = {
                str(player_id).strip()
                for player_id, _transaction_id in session_tx.get(pk, [])
                if pd.notna(player_id) and str(player_id).strip()
            }
            for player_id in players:
                counts = rounds_per_session_per_speler.setdefault(player_id, [])
                if pd.notna(round_count):
                    counts.append(int(round_count))

    # Calculate median per player
    records = []
    for player_id, session_counts in rounds_per_session_per_speler.items():
        if session_counts:
            median_rounds = float(np.median(session_counts))
        else:
            median_rounds = np.nan

        records.append({
            "Player_Profile_ID": player_id,
            "f31_median_rounds_per_session": median_rounds
        })

    if records:
        result = pd.DataFrame.from_records(records)
    else:
        print("   ⚠️  F31: Geen game session data gevonden voor deze operator — lege DataFrame geretourneerd.")
        if logger:
            logger.warning("⚠️  F31: Geen game session data gevonden voor deze operator — lege DataFrame geretourneerd.")
        result = pd.DataFrame(columns=["Player_Profile_ID", "f31_median_rounds_per_session"])

    if logger:
        logger.info(f"✅ F31 Median Rounds per Session klaar: {len(result):,} spelers")
        if len(result) > 0:
            valid = result[result["f31_median_rounds_per_session"].notna()]
            if len(valid) > 0:
                logger.info(f"   Gemiddelde van spelermedianen: {valid['f31_median_rounds_per_session'].mean():.1f} rounds/session")

    return result


# ------------------------------
# F32: Game Types Count (with join to WOK_Game.Game_Type)
# ------------------------------
def f32_game_types_count(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F32: Game Types Count

    Count number of distinct Game_Type values played by each player within the reference period.
    Join logic:
      - Build mapping Game_ID -> Game_Type from WOK_Game
      - From WOK_Game_Session, use Game_Transactions JSON to get Player_Profile_IDs per session
      - For each (player, session), add the mapped Game_Type (fallback: Game_ID if type unknown)

    Output:
      - f32_game_types_count: int (>=0). Anchored on F1 players => players with no sessions get 0.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f32_game_types_count")
        logger.info("▶ START F32: Game Types Count")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # ---- tijdspad ----
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    # ---- anchor on F1 so "no sessions" -> 0 ----
    f1 = f1_active_days(tables, x_tijdspad=x_tijdspad, chunksize=chunksize, log_path=None, verbose=verbose)
    f1_players = set(f1["Player_Profile_ID"].astype(str).tolist()) if not f1.empty else set()

    session_paths = tables.get("WOK_Game_Session")
    if not session_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f32_game_types_count"])

    # ---- build Game_ID -> Game_Type mapping ----
    game_paths = tables.get("WOK_Game") or []
    game_type_by_id: Dict[str, str] = {}

    if game_paths:
        for gdf in iter_csv_chunks(
            paths=game_paths,
            usecols=["Game_ID", "Game_Type"],
            chunksize=chunksize,
            verbose=verbose,
        ):
            if gdf.empty:
                continue
            gdf = gdf[gdf["Game_ID"].notna()].copy()
            if gdf.empty:
                continue

            # keep latest non-null type seen (doesn't really matter; Game table is usually unique per ID)
            gid_col = gdf.columns.get_loc("Game_ID")
            gty_col = gdf.columns.get_loc("Game_Type") if "Game_Type" in gdf.columns else None

            for i in range(len(gdf)):
                gid = gdf.iat[i, gid_col]
                if gid is None or (isinstance(gid, float) and np.isnan(gid)):
                    continue
                gid = str(gid)

                gty = None
                if gty_col is not None:
                    gty = gdf.iat[i, gty_col]
                if gty is None or (isinstance(gty, float) and np.isnan(gty)):
                    continue

                gty_s = str(gty).strip()
                if gty_s:
                    game_type_by_id[gid] = gty_s

    # ---- per player: set of distinct game types ----
    types_per_speler: Dict[str, set] = {}

    # organisatie-relationeel: transacties per sessie uit WOK_Game_Session_Transaction (join op pk_id).
    session_tx = build_session_tx_map(tables.get("WOK_Game_Session_Transaction"), chunksize=chunksize)

    need_cols = ["pk_id", "Game_Session_Start_Datetime", "Game_ID"]

    for df in iter_csv_chunks(
        paths=session_paths,
        usecols=need_cols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        # Time filtering on session start
        if start_datum is not None:
            ts = pd.to_datetime(df["Game_Session_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask = (ts >= start_datum) & (ts < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()
            if df.empty:
                continue

        col_idx_tx   = df.columns.get_loc("pk_id")
        col_idx_game = df.columns.get_loc("Game_ID")

        for i in range(len(df)):
            game_id = df.iat[i, col_idx_game]
            if game_id is None or (isinstance(game_id, float) and np.isnan(game_id)):
                continue
            game_id = str(game_id)

            game_type = game_type_by_id.get(game_id) or game_id  # fallback if unknown

            pk = df.iat[i, col_idx_tx]

            try:
                for player_id, _tx_id in session_tx.get(pk, []):
                    if player_id is None:
                        continue
                    pid = str(player_id)

                    if f1_players and pid not in f1_players:
                        continue

                    s = types_per_speler.get(pid)
                    if s is None:
                        s = set()
                        types_per_speler[pid] = s
                    s.add(game_type)
            except Exception:
                continue

    # ---- build output ----
    records = []
    if f1_players:
        for pid in f1_players:
            records.append({
                "Player_Profile_ID": pid,
                "f32_game_types_count": len(types_per_speler.get(pid, set())),
            })
    else:
        for pid, s in types_per_speler.items():
            records.append({
                "Player_Profile_ID": pid,
                "f32_game_types_count": len(s),
            })

    if records:
        result = pd.DataFrame.from_records(records)
    else:
        print("   ⚠️  F32: Geen game session data gevonden voor deze operator — lege DataFrame geretourneerd.")
        if logger:
            logger.warning("⚠️  F32: Geen game session data gevonden voor deze operator — lege DataFrame geretourneerd.")
        result = pd.DataFrame(columns=["Player_Profile_ID", "f32_game_types_count"])

    if logger:
        logger.info(f"✅ F32 Game Types Count klaar: {len(result):,} spelers")
        if len(result) > 0:
            logger.info(f"   Gemiddeld aantal game types: {result['f32_game_types_count'].mean():.2f}")

    return result

# ------------------------------------------------------------
# F33–F38 (proxy): 70% segment flags using ONLY Dutch CDB Game_Type buckets
# ------------------------------------------------------------
def f33_f34_f35_f36_f37_f38_segments_cdb6(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build 6 binary flags per player: 1 iff >=70% of wagered amounts are in that bucket.

    Buckets we use (CDB-native):
      - 70_procent_weddenschappen  (proxy for F33)
      - 70_procent_OTHER          (proxy for F34)
      - 70_procent_CASINO         (proxy for F35)
      - 70_procent_VIRTUAL_SPORTS (proxy for F36)
      - 70_procent_SLOTS          (proxy for F37)
      - 70_procent_BINGO          (proxy for F38)

    "Wagered amount" definition:
      - all SUCCESSFUL negative interactions (Transaction_Amount < 0),
        wager = abs(amount)
      - STAKE amounts are forced negative (one-time warning if flipped)

    Classification:
      - If Transaction_ID is referenced by WOK_Bet.Bet_Transactions -> "weddenschappen"
      - Else if Transaction_ID is referenced by WOK_Game_Session.Game_Transactions:
            map Game_ID -> Game_Type via WOK_Game, bucket = that Game_Type (SLOTS/CASINO/BINGO/VIRTUAL_SPORTS/OTHER)
      - Else -> bucket "OTHER" (keeps totals consistent)

    Output:
      Player_Profile_ID plus the 6 flags (0/1)

    Notes:
      - Filter is [start, end) with no +1 day logic.
      - Anchored on F1 players if available; otherwise on players observed in transactions.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f33_f34_f35_f36_f37_f38_segments_cdb6")
        logger.info("▶ START F33–F38 (CDB6): 70% segment flags")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # ---- tijdspad ----
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    # ---- anchor on F1 players (if present) ----
    f1 = f1_active_days(tables, x_tijdspad=x_tijdspad, chunksize=chunksize, log_path=None, verbose=verbose)
    f1_players = set(f1["Player_Profile_ID"].astype(str).tolist()) if not f1.empty else set()

    # ------------------------------------------------------------
    # 1) BETTING txids: WOK_Bet.Bet_Transactions
    # ------------------------------------------------------------
    bet_txids: set[str] = set()
    bet_paths = tables.get("WOK_Bet") or []
    if bet_paths:
        # organisatie-relationeel: tx-refs per bet via WOK_Bet_Transaction (join op pk_id).
        _bet_tx_map = build_bet_tx_map(tables.get("WOK_Bet_Transaction"), chunksize=chunksize)
        for bdf in iter_csv_chunks(
            paths=bet_paths,
            usecols=["pk_id", "Bet_Start_Datetime"],
            chunksize=chunksize,
            verbose=verbose,
        ):
            if bdf.empty:
                continue

            if start_datum is not None:
                ts = pd.to_datetime(bdf["Bet_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
                mask = (ts >= start_datum) & (ts < eind_datum)
                if not mask.any():
                    continue
                bdf = bdf.loc[mask].copy()
                if bdf.empty:
                    continue

            col_pk = bdf.columns.get_loc("pk_id")
            for i in range(len(bdf)):
                bpk = bdf.iat[i, col_pk]
                for _pid, txid in _bet_tx_map.get(bpk, []):
                    if txid is not None and not (isinstance(txid, float) and pd.isna(txid)):
                        bet_txids.add(str(txid))

    # ------------------------------------------------------------
    # 2) Game_ID -> Game_Type map (WOK_Game)
    # ------------------------------------------------------------
    game_type_by_id: Dict[str, str] = {}
    game_paths = tables.get("WOK_Game") or []
    if game_paths:
        for gdf in iter_csv_chunks(
            paths=game_paths,
            usecols=["Game_ID", "Game_Type"],
            chunksize=chunksize,
            verbose=verbose,
        ):
            if gdf.empty:
                continue
            gdf = gdf[gdf["Game_ID"].notna()].copy()
            if gdf.empty:
                continue

            col_gid = gdf.columns.get_loc("Game_ID")
            col_gty = gdf.columns.get_loc("Game_Type") if "Game_Type" in gdf.columns else None
            if col_gty is None:
                continue

            for i in range(len(gdf)):
                gid = gdf.iat[i, col_gid]
                gty = gdf.iat[i, col_gty]
                if gid is None or (isinstance(gid, float) and np.isnan(gid)):
                    continue
                if gty is None or (isinstance(gty, float) and np.isnan(gty)):
                    continue
                gid_s = str(gid)
                gty_s = str(gty).strip()
                if gid_s not in game_type_by_id:
                    game_type_by_id[gid_s] = gty_s
                else:
                    # keep first; usually stable
                    pass

    # ------------------------------------------------------------
    # 3) Transaction_ID -> Game_Type bucket (WOK_Game_Session)
    # ------------------------------------------------------------
    txid_to_gametype: Dict[str, str] = {}
    session_paths = tables.get("WOK_Game_Session") or []
    # organisatie-relationeel: transacties per sessie uit WOK_Game_Session_Transaction (join op pk_id).
    session_tx = build_session_tx_map(tables.get("WOK_Game_Session_Transaction"), chunksize=chunksize)
    if session_paths:
        for sdf in iter_csv_chunks(
            paths=session_paths,
            usecols=["Game_Session_Start_Datetime", "Game_ID", "pk_id"],
            chunksize=chunksize,
            verbose=verbose,
        ):
            if sdf.empty:
                continue

            if start_datum is not None:
                ts = pd.to_datetime(sdf["Game_Session_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
                mask = (ts >= start_datum) & (ts < eind_datum)
                if not mask.any():
                    continue
                sdf = sdf.loc[mask].copy()
                if sdf.empty:
                    continue

            col_gid = sdf.columns.get_loc("Game_ID")
            col_gtx = sdf.columns.get_loc("pk_id")

            for i in range(len(sdf)):
                gid = sdf.iat[i, col_gid]
                if gid is None or (isinstance(gid, float) and np.isnan(gid)):
                    continue
                gid_s = str(gid)
                gtype = game_type_by_id.get(gid_s, "OTHER")
                if gtype not in {"SLOTS", "CASINO", "BINGO", "VIRTUAL_SPORTS", "OTHER"}:
                    gtype = "OTHER"

                pk = sdf.iat[i, col_gtx]

                try:
                    for _pid, txid in session_tx.get(pk, []):
                        if txid is None:
                            continue
                        txid_to_gametype[str(txid)] = gtype
                except Exception:
                    continue

    # ------------------------------------------------------------
    # 4) Stream account tx and accumulate wager sums per bucket
    # ------------------------------------------------------------
    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=[
            "Player_Profile_ID",
            "70_procent_weddenschappen",
            "70_procent_SLOTS",
            "70_procent_CASINO",
            "70_procent_BINGO",
            "70_procent_VIRTUAL_SPORTS",
            "70_procent_OTHER",
        ])

    # sums
    total_wager: Dict[str, float] = {}
    sum_wedd: Dict[str, float] = {}
    sum_slots: Dict[str, float] = {}
    sum_casino: Dict[str, float] = {}
    sum_bingo: Dict[str, float] = {}
    sum_vs: Dict[str, float] = {}
    sum_other: Dict[str, float] = {}

    coerced_stake_warned = False

    for tdf in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID","Transaction_ID","Transaction_Datetime","Transaction_Amount","Transaction_Type","Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        tdf = tdf[tdf["Player_Profile_ID"].notna()].copy()
        if tdf.empty:
            continue

        tdf = tdf[tdf["Transaction_Status"] == "SUCCESSFUL"]
        if tdf.empty:
            continue

        ts = pd.to_datetime(tdf["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        tdf["ts"] = ts
        tdf = tdf[tdf["ts"].notna()]
        if tdf.empty:
            continue

        if start_datum is not None:
            mask = (tdf["ts"] >= start_datum) & (tdf["ts"] < eind_datum)
            if not mask.any():
                continue
            tdf = tdf.loc[mask].copy()
            if tdf.empty:
                continue

        tdf["amount"] = pd.to_numeric(tdf["Transaction_Amount"], errors="coerce")
        tdf = tdf[tdf["amount"].notna()]
        if tdf.empty:
            continue

        # enforce STAKE negative
        is_stake = tdf["Transaction_Type"].astype(str).eq("STAKE")
        to_flip = is_stake & (tdf["amount"] > 0)
        if to_flip.any():
            tdf.loc[to_flip, "amount"] = -tdf.loc[to_flip, "amount"]
            if logger and not coerced_stake_warned:
                logger.warning("⚠️ Coerced positive STAKE amounts to negative (one-time warning).")
            coerced_stake_warned = True

        # keep only negative interactions as wager
        tdf = tdf[tdf["amount"] < 0]
        if tdf.empty:
            continue

        pid_col = tdf.columns.get_loc("Player_Profile_ID")
        tid_col = tdf.columns.get_loc("Transaction_ID")
        amt_col = tdf.columns.get_loc("amount")

        for i in range(len(tdf)):
            pid = str(tdf.iat[i, pid_col])
            if f1_players and pid not in f1_players:
                continue

            txid = tdf.iat[i, tid_col]
            txid_s = str(txid) if txid is not None and not (isinstance(txid, float) and np.isnan(txid)) else ""

            wager = float(-tdf.iat[i, amt_col])  # abs(negative)
            total_wager[pid] = total_wager.get(pid, 0.0) + wager

            # bucket
            if txid_s and txid_s in bet_txids:
                sum_wedd[pid] = sum_wedd.get(pid, 0.0) + wager
            else:
                gtype = txid_to_gametype.get(txid_s, "OTHER") if txid_s else "OTHER"
                if gtype == "SLOTS":
                    sum_slots[pid] = sum_slots.get(pid, 0.0) + wager
                elif gtype == "CASINO":
                    sum_casino[pid] = sum_casino.get(pid, 0.0) + wager
                elif gtype == "BINGO":
                    sum_bingo[pid] = sum_bingo.get(pid, 0.0) + wager
                elif gtype == "VIRTUAL_SPORTS":
                    sum_vs[pid] = sum_vs.get(pid, 0.0) + wager
                else:
                    sum_other[pid] = sum_other.get(pid, 0.0) + wager

    # ------------------------------------------------------------
    # 5) Flags: >=70%
    # ------------------------------------------------------------
    def _flag(seg_sum: float, tot: float) -> int:
        if tot <= 0:
            return 0
        return 1 if (seg_sum / tot) >= 0.70 else 0

    players_out = sorted(f1_players) if f1_players else sorted(total_wager.keys())

    records = []
    for pid in players_out:
        tot = total_wager.get(pid, 0.0)
        records.append({
            "Player_Profile_ID": pid,

            # Rough mapping to your numbering request:
            #   weddenschappen ~ 33
            #   OTHER          ~ 34
            #   CASINO         ~ 35
            #   VIRTUAL_SPORTS ~ 36
            #   SLOTS          ~ 37
            #   BINGO          ~ 38
            "70_procent_weddenschappen": _flag(sum_wedd.get(pid, 0.0), tot),
            "70_procent_SLOTS":          _flag(sum_slots.get(pid, 0.0), tot),
            "70_procent_CASINO":         _flag(sum_casino.get(pid, 0.0), tot),
            "70_procent_BINGO":          _flag(sum_bingo.get(pid, 0.0), tot),
            "70_procent_VIRTUAL_SPORTS": _flag(sum_vs.get(pid, 0.0), tot),
            "70_procent_OTHER":          _flag(sum_other.get(pid, 0.0), tot),
        })

    if records:
        out = pd.DataFrame.from_records(records)
    else:
        print("   ⚠️  F33-F38: Geen bet/session data gevonden voor deze operator — lege DataFrame geretourneerd.")
        if logger:
            logger.warning("⚠️  F33-F38: Geen bet/session data gevonden voor deze operator — lege DataFrame geretourneerd.")
        out = pd.DataFrame(columns=[
            "Player_Profile_ID",
            "70_procent_weddenschappen", "70_procent_SLOTS", "70_procent_CASINO",
            "70_procent_BINGO", "70_procent_VIRTUAL_SPORTS", "70_procent_OTHER",
        ])

    if logger:
        logger.info(f"✅ Segments (CDB6) klaar: {len(out):,} spelers")
        if len(out) > 0:
            logger.info(
                "   counts>=70%: "
                f"WEDD={out['70_procent_weddenschappen'].sum():,} "
                f"SLOTS={out['70_procent_SLOTS'].sum():,} "
                f"CASINO={out['70_procent_CASINO'].sum():,} "
                f"BINGO={out['70_procent_BINGO'].sum():,} "
                f"VSPORTS={out['70_procent_VIRTUAL_SPORTS'].sum():,} "
                f"OTHER={out['70_procent_OTHER'].sum():,}"
            )

    return out


# ------------------------------------------------------------
# F39 dominant segment share (proxy): share of stakes in dominant segment
# ------------------------------------------------------------

def f39_dominant_segment_share(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F39: Dominant Segment Share

    Calculate the proportion of player's stakes concentrated in their top game segment.

    Input:
        - WOK_Bet: Sports betting stakes
        - WOK_Game_Session: Casino game stakes
        - WOK_Player_Account_Transaction: Total stakes

    Output:
        - f39_dominant_segment_share: Float (0-1, share of stakes in dominant segment)
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f39_dominant_segment_share")
        logger.info("▶ START F39: Dominant Segment Share")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]}")
    else:
        logger = None

    # Parse tijdspad
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum = None

    # Track stakes by segment: {player_id: {"sports": float, "casino": float}}
    stakes_by_segment: Dict[str, Dict[str, float]] = {}

    # Get sports stakes from WOK_Bet
    # organisatie-relationeel: speler via WOK_Bet_Transaction (join op pk_id), stake op de bet-rij zelf.
    bet_paths = tables.get("WOK_Bet")
    if bet_paths:
        _bet_tx_map = build_bet_tx_map(tables.get("WOK_Bet_Transaction"), chunksize=chunksize)
        for df in iter_csv_chunks(
            paths=bet_paths,
            usecols=["pk_id", "Bet_Total_Stake", "Bet_Start_Datetime"],
            chunksize=chunksize,
            verbose=verbose,
        ):
            if df.empty:
                continue

            # Time filtering
            if start_datum is not None:
                df["timestamp"] = pd.to_datetime(df["Bet_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
                mask_periode = (df["timestamp"] >= start_datum) & (df["timestamp"] < eind_datum)
                if not mask_periode.any():
                    continue
                df = df.loc[mask_periode].copy()

            col_idx_pk = df.columns.get_loc("pk_id")
            col_idx_stake = df.columns.get_loc("Bet_Total_Stake")

            for rij_index in range(len(df)):
                pk = df.iat[rij_index, col_idx_pk]
                stake = pd.to_numeric(df.iat[rij_index, col_idx_stake], errors="coerce")

                if pd.isna(stake) or stake <= 0:
                    continue

                player_ids = [str(p) for p, _t in _bet_tx_map.get(pk, [])
                              if p is not None and not (isinstance(p, float) and pd.isna(p))]

                if player_ids:
                    player_id = player_ids[0]
                    if player_id not in stakes_by_segment:
                        stakes_by_segment[player_id] = {"sports": 0.0, "casino": 0.0}
                    stakes_by_segment[player_id]["sports"] += float(stake)

    # Get casino stakes from WOK_Game_Session (approximate from game transactions)
    # organisatie-relationeel: transacties per sessie uit WOK_Game_Session_Transaction (join op pk_id).
    session_paths = tables.get("WOK_Game_Session")
    if session_paths:
        _session_tx_map = build_session_tx_map(tables.get("WOK_Game_Session_Transaction"), chunksize=chunksize)
        for df in iter_csv_chunks(
            paths=session_paths,
            usecols=["pk_id", "Game_Session_Start_Datetime"],
            chunksize=chunksize,
            verbose=verbose,
        ):
            if df.empty:
                continue

            # Time filtering
            if start_datum is not None:
                df["timestamp"] = pd.to_datetime(df["Game_Session_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
                mask_periode = (df["timestamp"] >= start_datum) & (df["timestamp"] < eind_datum)
                if not mask_periode.any():
                    continue
                df = df.loc[mask_periode].copy()

            col_idx_pk = df.columns.get_loc("pk_id")

            for rij_index in range(len(df)):
                spk = df.iat[rij_index, col_idx_pk]

                # Count transactions as proxy for casino activity
                txs = _session_tx_map.get(spk, [])
                tx_count = len(txs)
                player_id_in_session = None
                for p, _tx_id in txs:
                    if p is not None and not (isinstance(p, float) and pd.isna(p)):
                        player_id_in_session = str(p)

                if player_id_in_session and tx_count > 0:
                    if player_id_in_session not in stakes_by_segment:
                        stakes_by_segment[player_id_in_session] = {"sports": 0.0, "casino": 0.0}
                    # Approximate casino stake (we don't have exact amounts, use transaction count as proxy)
                    stakes_by_segment[player_id_in_session]["casino"] += float(tx_count)

    # Calculate dominant share
    records = []
    for player_id, segments in stakes_by_segment.items():
        total = segments["sports"] + segments["casino"]
        if total > 0:
            max_segment = max(segments["sports"], segments["casino"])
            share = max_segment / total
        else:
            share = 0.0

        records.append({
            "Player_Profile_ID": player_id,
            "f39_dominant_segment_share": share
        })

    if records:
        result = pd.DataFrame.from_records(records)
    else:
        result = pd.DataFrame(columns=["Player_Profile_ID", "f39_dominant_segment_share"])

    if logger:
        logger.info(f"✅ F39 Dominant Segment Share klaar: {len(result):,} spelers")
        if len(result) > 0:
            logger.info(f"   Gemiddelde dominant share: {result['f39_dominant_segment_share'].mean():.2f}")

    return result

# ------------------------------
# F40: Different Products per Active Day
# ------------------------------

def f40_products_per_active_day(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F40:
    Number of different products played by the player under sessions of
    other games or pre-drawn games, divided by F1.

    Mapping to CDB:
      - Sessions → WOK_Game_Session
      - Product  → Game_Commercial_Name via WOK_Game
      - Excludes betting (only game sessions)
      - Output = unique products / F1_active_days
      - NaN if no session played

    Output column:
      - f40_products_per_active_day
    """

    if log_path:
        logger = _setup_feature_logger(log_path, "f40_products_per_active_day")
        logger.info("▶ START F40: Different Products per Active Day")
    else:
        logger = None

    # ---- tijdspad ----
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    # ---- F1 (denominator) ----
    f1 = f1_active_days(tables, x_tijdspad=x_tijdspad, chunksize=chunksize, log_path=None, verbose=verbose)
    if f1.empty:
        return pd.DataFrame(columns=["Player_Profile_ID", "f40_products_per_active_day"])

    f1_dict = f1.set_index("Player_Profile_ID")["f1_active_days"].to_dict()

    # ---- Game_ID → Game_Commercial_Name mapping ----
    game_paths = tables.get("WOK_Game")
    if not game_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f40_products_per_active_day"])

    game_id_to_name: Dict[str, str] = {}

    for df in iter_csv_chunks(
        paths=game_paths,
        usecols=["Game_ID", "Game_Commercial_Name"],
        chunksize=chunksize,
        verbose=verbose,
    ):

        df = df[df["Game_ID"].notna() & df["Game_Commercial_Name"].notna()]
        for _, r in df.iterrows():
            game_id_to_name[str(r["Game_ID"])] = str(r["Game_Commercial_Name"])

    # ---- Track unique products per player ----
    session_paths = tables.get("WOK_Game_Session")
    if not session_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f40_products_per_active_day"])

    products_per_speler: Dict[str, set] = {}
    spelers_met_sessie: set = set()

    # organisatie-relationeel: transacties per sessie uit WOK_Game_Session_Transaction (join op pk_id).
    session_tx = build_session_tx_map(tables.get("WOK_Game_Session_Transaction"), chunksize=chunksize)

    for df in iter_csv_chunks(
        paths=session_paths,
        usecols=["pk_id", "Game_Session_Start_Datetime", "Game_ID"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        # ---- tijdsfilter ----
        if start_datum is not None:
            ts = pd.to_datetime(df["Game_Session_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask = (ts >= start_datum) & (ts < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask]

        idx_tx   = df.columns.get_loc("pk_id")
        idx_game = df.columns.get_loc("Game_ID")

        for i in range(len(df)):
            game_id = df.iat[i, idx_game]
            if pd.isna(game_id):
                continue

            product_name = game_id_to_name.get(str(game_id))
            if not product_name:
                continue

            # welke spelers zitten in deze sessie? (relationele join i.p.v. JSON)
            for player_id, _ in session_tx.get(df.iat[i, idx_tx], []):
                spelers_met_sessie.add(player_id)

                if player_id not in products_per_speler:
                    products_per_speler[player_id] = set()

                products_per_speler[player_id].add(product_name)

    # ---- Resultaat bouwen ----
    records = []

    for pid, active_days in f1_dict.items():
        if pid not in spelers_met_sessie:
            value = np.nan  # geen sessie gespeeld → N/A
        else:
            n_products = len(products_per_speler.get(pid, set()))
            value = (n_products / active_days) if active_days > 0 else np.nan

        records.append({
            "Player_Profile_ID": pid,
            "f40_products_per_active_day": value
        })

    if records:
        result = pd.DataFrame.from_records(records)
    else:
        print("   ⚠️  F40: Geen game session data gevonden voor deze operator — lege DataFrame geretourneerd.")
        if logger:
            logger.warning("⚠️  F40: Geen game session data gevonden voor deze operator — lege DataFrame geretourneerd.")
        result = pd.DataFrame(columns=["Player_Profile_ID", "f40_products_per_active_day"])

    if logger:
        logger.info(f"✅ F40 klaar: {len(result):,} spelers")
        na = result["f40_products_per_active_day"].isna().sum()
        logger.info(f"   N/A (geen sessies): {na:,}")

    return result


# ------------------------------
# F41: Heavy-play hours count (SPEC CORRECT: >= 1/48 of interactions)
# ------------------------------

def f41_heavy_play_hours_count(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F41 (spec-correct):
    Number of hours (0..23) such that:
      - at least one interaction occurred in that hour, AND
      - that hour accounts for >= 1/48 of the player's total interactions
        during the specified period.

    "Interactions" are defined here as Transaction_IDs referenced by:
      - WOK_Game_Session.Game_Transactions
      - WOK_Bet.Bet_Transactions

    Hour assignment uses TRUE Transaction_Datetime from WOK_Player_Account_Transaction
    (so we don't bin everything by session/bet start time).

    Output:
      - f41_heavy_play_hours_count: int (0..24), 0 if no interactions.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f41_heavy_play_hours_count")
        logger.info("▶ START F41: Heavy-play hours count (>=1/48)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # ---- tijdspad ----
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    session_paths = tables.get("WOK_Game_Session") or []
    bet_paths     = tables.get("WOK_Bet") or []
    acct_paths    = tables.get("WOK_Player_Account_Transaction") or []

    if not acct_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f41_heavy_play_hours_count"])

    # ---------------------------------------------------------
    # 1) Collect relevant Transaction_IDs from sessions + bets
    # ---------------------------------------------------------
    wanted_txids: set[str] = set()

    if session_paths:
        # organisatie-relationeel: tx-refs per sessie via WOK_Game_Session_Transaction (join op pk_id).
        _session_tx_map = build_session_tx_map(tables.get("WOK_Game_Session_Transaction"), chunksize=chunksize)
        for df in iter_csv_chunks(
            paths=session_paths,
            usecols=["pk_id", "Game_Session_Start_Datetime"],
            chunksize=chunksize,
            verbose=verbose,
        ):
            if df.empty:
                continue

            # cheap prefilter on session start (optional; real filtering happens on tx datetime)
            if start_datum is not None:
                ts = pd.to_datetime(df["Game_Session_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
                m = (ts >= start_datum) & (ts < eind_datum)
                if not m.any():
                    continue
                df = df.loc[m].copy()

            idx_pk = df.columns.get_loc("pk_id")
            for i in range(len(df)):
                spk = df.iat[i, idx_pk]
                for _pid, txid in _session_tx_map.get(spk, []):
                    if txid is not None and not (isinstance(txid, float) and pd.isna(txid)):
                        wanted_txids.add(str(txid))

    if bet_paths:
        # organisatie-relationeel: tx-refs per bet via WOK_Bet_Transaction (join op pk_id).
        _bet_tx_map = build_bet_tx_map(tables.get("WOK_Bet_Transaction"), chunksize=chunksize)
        for df in iter_csv_chunks(
            paths=bet_paths,
            usecols=["pk_id", "Bet_Start_Datetime"],
            chunksize=chunksize,
            verbose=verbose,
        ):
            if df.empty:
                continue

            # cheap prefilter on bet start (optional; real filtering happens on tx datetime)
            if start_datum is not None:
                ts = pd.to_datetime(df["Bet_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
                m = (ts >= start_datum) & (ts < eind_datum)
                if not m.any():
                    continue
                df = df.loc[m].copy()

            idx_pk = df.columns.get_loc("pk_id")
            for i in range(len(df)):
                bpk = df.iat[i, idx_pk]
                for _pid, txid in _bet_tx_map.get(bpk, []):
                    if txid is not None and not (isinstance(txid, float) and pd.isna(txid)):
                        wanted_txids.add(str(txid))

    if not wanted_txids:
        # no interactions observed -> return empty (consistent with your other features)
        return pd.DataFrame(columns=["Player_Profile_ID", "f41_heavy_play_hours_count"])

    # ---------------------------------------------------------
    # 2) Stream account transactions and bin (pid, hour)
    #    Only for Transaction_IDs we saw in sessions/bets
    # ---------------------------------------------------------
    total_per_pid: Dict[str, int] = {}
    hour_counts: Dict[str, np.ndarray] = {}

    for df in iter_csv_chunks(
        paths=acct_paths,
        usecols=["Transaction_ID", "Player_Profile_ID", "Transaction_Datetime", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        df = df[df["Transaction_Status"] == "SUCCESSFUL"]
        if df.empty:
            continue

        df = df[df["Transaction_ID"].notna() & df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        df["Transaction_ID"] = df["Transaction_ID"].astype(str)
        m = df["Transaction_ID"].isin(wanted_txids)
        if not m.any():
            continue
        df = df.loc[m].copy()

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            m2 = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not m2.any():
                continue
            df = df.loc[m2].copy()

        df["hour"] = df["ts"].dt.hour.astype("int16")

        grp = df.groupby(["Player_Profile_ID", "hour"]).size()
        for (pid, hour), c in grp.items():
            pid = str(pid)
            h = int(hour)
            if pid not in hour_counts:
                hour_counts[pid] = np.zeros(24, dtype=np.int64)
                total_per_pid[pid] = 0
            hour_counts[pid][h] += int(c)
            total_per_pid[pid] += int(c)

    # ---------------------------------------------------------
    # 3) Apply 1/48 threshold and count qualifying hours
    # ---------------------------------------------------------
    records = []
    for pid, total in total_per_pid.items():
        if total <= 0:
            val = 0
        else:
            thr = total / 48.0
            arr = hour_counts.get(pid)
            val = int(np.sum(arr >= thr)) if arr is not None else 0
        records.append({"Player_Profile_ID": pid, "f41_heavy_play_hours_count": val})

    result = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "f41_heavy_play_hours_count"]
    )

    if logger:
        logger.info(f"✅ F41 klaar: {len(result):,} spelers")
        if len(result) > 0:
            logger.info(f"   Gemiddeld #uren: {result['f41_heavy_play_hours_count'].mean():.2f}")

    return result
# ------------------------------
# F42: morning interaction percentage
# ------------------------------

def f42_morning_interaction_percentage(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F42 (spec):
    Fraction of interactions during the specified period that occur in time slot:
      08:00:00 to 15:59:59  =>  8 <= hour < 16

    Interactions are approximated as:
      - Transaction_IDs referenced in WOK_Game_Session.Game_Transactions
      - Transaction_IDs referenced in WOK_Bet.Bet_Transactions

    Output:
      - f42_morning_interaction_percentage: float in [0,1], NaN if total interactions == 0
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f42_morning_interaction_percentage")
        logger.info("▶ START F42: Morning interaction fraction (08:00-15:59)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    total_per_pid: Dict[str, int] = {}
    morning_per_pid: Dict[str, int] = {}

    # --- Game sessions ---
    # organisatie-relationeel: tx-refs per sessie via WOK_Game_Session_Transaction (join op pk_id).
    session_paths = tables.get("WOK_Game_Session") or []
    if session_paths:
        _session_tx_map = build_session_tx_map(tables.get("WOK_Game_Session_Transaction"), chunksize=chunksize)
        for df in iter_csv_chunks(
            paths=session_paths,
            usecols=["pk_id", "Game_Session_Start_Datetime"],
            chunksize=chunksize,
            verbose=verbose,
        ):
            if df.empty:
                continue

            ts = pd.to_datetime(df["Game_Session_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            df = df[ts.notna()].copy()
            if df.empty:
                continue
            df["ts"] = ts[ts.notna()].values

            if start_datum is not None:
                m = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
                if not m.any():
                    continue
                df = df.loc[m].copy()

            df["hour"] = df["ts"].dt.hour.astype("int16")

            idx_pk = df.columns.get_loc("pk_id")
            idx_hr = df.columns.get_loc("hour")

            for i in range(len(df)):
                spk = df.iat[i, idx_pk]
                hour = int(df.iat[i, idx_hr])

                # count txids per player (within this session row)
                per_player_counts: Dict[str, int] = {}
                for pid_raw, _txid in _session_tx_map.get(spk, []):
                    if pid_raw is None or (isinstance(pid_raw, float) and pd.isna(pid_raw)):
                        continue
                    pid = str(pid_raw)
                    per_player_counts[pid] = per_player_counts.get(pid, 0) + 1

                if not per_player_counts:
                    continue

                is_morning = (8 <= hour < 16)
                for pid, c in per_player_counts.items():
                    total_per_pid[pid] = total_per_pid.get(pid, 0) + c
                    if is_morning:
                        morning_per_pid[pid] = morning_per_pid.get(pid, 0) + c

    # --- Bets ---
    # organisatie-relationeel: tx-refs per bet via WOK_Bet_Transaction (join op pk_id).
    bet_paths = tables.get("WOK_Bet") or []
    if bet_paths:
        _bet_tx_map = build_bet_tx_map(tables.get("WOK_Bet_Transaction"), chunksize=chunksize)
        for df in iter_csv_chunks(
            paths=bet_paths,
            usecols=["pk_id", "Bet_Start_Datetime"],
            chunksize=chunksize,
            verbose=verbose,
        ):
            if df.empty:
                continue

            ts = pd.to_datetime(df["Bet_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            df = df[ts.notna()].copy()
            if df.empty:
                continue
            df["ts"] = ts[ts.notna()].values

            if start_datum is not None:
                m = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
                if not m.any():
                    continue
                df = df.loc[m].copy()

            df["hour"] = df["ts"].dt.hour.astype("int16")

            idx_pk = df.columns.get_loc("pk_id")
            idx_hr = df.columns.get_loc("hour")

            for i in range(len(df)):
                bpk = df.iat[i, idx_pk]
                hour = int(df.iat[i, idx_hr])

                # count txids per player (within this bet row)
                per_player_counts: Dict[str, int] = {}
                for pid_raw, _txid in _bet_tx_map.get(bpk, []):
                    if pid_raw is None or (isinstance(pid_raw, float) and pd.isna(pid_raw)):
                        continue
                    pid = str(pid_raw)
                    per_player_counts[pid] = per_player_counts.get(pid, 0) + 1

                if not per_player_counts:
                    continue

                is_morning = (8 <= hour < 16)
                for pid, c in per_player_counts.items():
                    total_per_pid[pid] = total_per_pid.get(pid, 0) + c
                    if is_morning:
                        morning_per_pid[pid] = morning_per_pid.get(pid, 0) + c

    # --- output ---
    pids = sorted(total_per_pid.keys())
    records = []
    for pid in pids:
        tot = total_per_pid.get(pid, 0)
        if tot <= 0:
            val = np.nan
        else:
            val = float(morning_per_pid.get(pid, 0)) / float(tot)
        records.append({"Player_Profile_ID": pid, "f42_morning_interaction_percentage": val})

    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "f42_morning_interaction_percentage"]
    )

    if logger:
        logger.info(f"✅ F42 klaar: {len(out):,} spelers")
    return out

# ------------------------------
# F42: evening interaction percentage
# ------------------------------


def f43_evening_interaction_percentage(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F43 (spec):
    Fraction of interactions during the specified period that occur in time slot:
      16:00:00 to 23:59:59  =>  16 <= hour < 24

    Interactions are approximated as:
      - Transaction_IDs referenced in WOK_Game_Session.Game_Transactions
      - Transaction_IDs referenced in WOK_Bet.Bet_Transactions

    Output:
      - f43_evening_interaction_percentage: float in [0,1], NaN if total interactions == 0
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f43_evening_interaction_percentage")
        logger.info("▶ START F43: Evening interaction fraction (16:00-23:59)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    total_per_pid: Dict[str, int] = {}
    evening_per_pid: Dict[str, int] = {}

    # --- Game sessions ---
    # organisatie-relationeel: tx-refs per sessie via WOK_Game_Session_Transaction (join op pk_id).
    session_paths = tables.get("WOK_Game_Session") or []
    if session_paths:
        _session_tx_map = build_session_tx_map(tables.get("WOK_Game_Session_Transaction"), chunksize=chunksize)
        for df in iter_csv_chunks(
            paths=session_paths,
            usecols=["pk_id", "Game_Session_Start_Datetime"],
            chunksize=chunksize,
            verbose=verbose,
        ):
            if df.empty:
                continue

            ts = pd.to_datetime(df["Game_Session_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            df = df[ts.notna()].copy()
            if df.empty:
                continue
            df["ts"] = ts[ts.notna()].values

            if start_datum is not None:
                m = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
                if not m.any():
                    continue
                df = df.loc[m].copy()

            df["hour"] = df["ts"].dt.hour.astype("int16")

            idx_pk = df.columns.get_loc("pk_id")
            idx_hr = df.columns.get_loc("hour")

            for i in range(len(df)):
                spk = df.iat[i, idx_pk]
                hour = int(df.iat[i, idx_hr])

                per_player_counts: Dict[str, int] = {}
                for pid_raw, _txid in _session_tx_map.get(spk, []):
                    if pid_raw is None or (isinstance(pid_raw, float) and pd.isna(pid_raw)):
                        continue
                    pid = str(pid_raw)
                    per_player_counts[pid] = per_player_counts.get(pid, 0) + 1

                if not per_player_counts:
                    continue

                is_evening = (16 <= hour < 24)
                for pid, c in per_player_counts.items():
                    total_per_pid[pid] = total_per_pid.get(pid, 0) + c
                    if is_evening:
                        evening_per_pid[pid] = evening_per_pid.get(pid, 0) + c

    # --- Bets ---
    # organisatie-relationeel: tx-refs per bet via WOK_Bet_Transaction (join op pk_id).
    bet_paths = tables.get("WOK_Bet") or []
    if bet_paths:
        _bet_tx_map = build_bet_tx_map(tables.get("WOK_Bet_Transaction"), chunksize=chunksize)
        for df in iter_csv_chunks(
            paths=bet_paths,
            usecols=["pk_id", "Bet_Start_Datetime"],
            chunksize=chunksize,
            verbose=verbose,
        ):
            if df.empty:
                continue

            ts = pd.to_datetime(df["Bet_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            df = df[ts.notna()].copy()
            if df.empty:
                continue
            df["ts"] = ts[ts.notna()].values

            if start_datum is not None:
                m = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
                if not m.any():
                    continue
                df = df.loc[m].copy()

            df["hour"] = df["ts"].dt.hour.astype("int16")

            idx_pk = df.columns.get_loc("pk_id")
            idx_hr = df.columns.get_loc("hour")

            for i in range(len(df)):
                bpk = df.iat[i, idx_pk]
                hour = int(df.iat[i, idx_hr])

                per_player_counts: Dict[str, int] = {}
                for pid_raw, _txid in _bet_tx_map.get(bpk, []):
                    if pid_raw is None or (isinstance(pid_raw, float) and pd.isna(pid_raw)):
                        continue
                    pid = str(pid_raw)
                    per_player_counts[pid] = per_player_counts.get(pid, 0) + 1

                if not per_player_counts:
                    continue

                is_evening = (16 <= hour < 24)
                for pid, c in per_player_counts.items():
                    total_per_pid[pid] = total_per_pid.get(pid, 0) + c
                    if is_evening:
                        evening_per_pid[pid] = evening_per_pid.get(pid, 0) + c

    # --- output ---
    pids = sorted(total_per_pid.keys())
    records = []
    for pid in pids:
        tot = total_per_pid.get(pid, 0)
        if tot <= 0:
            val = np.nan
        else:
            val = float(evening_per_pid.get(pid, 0)) / float(tot)
        records.append({"Player_Profile_ID": pid, "f43_evening_interaction_percentage": val})

    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "f43_evening_interaction_percentage"]
    )

    if logger:
        logger.info(f"✅ F43 klaar: {len(out):,} spelers")
    return out

# ------------------------------
# F44: morning stakes percentage
# ------------------------------

def f44_morning_stakes_percentage(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F44 (spec):
    Share (0..1) of amounts wagered (stakes) during the specified period that occur in:
      08:00:00 to 15:59:59  =>  8 <= hour < 16

    Uses WOK_Player_Account_Transaction:
      - Transaction_Type == "STAKE"
      - Transaction_Status == "SUCCESSFUL"
      - amount aggregated as abs(Transaction_Amount)

    Output:
      - f44_morning_stakes_percentage: float in [0,1], NaN if total stake == 0
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f44_morning_stakes_percentage")
        logger.info("▶ START F44: Morning stakes share (08:00-15:59)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    txn_paths = tables.get("WOK_Player_Account_Transaction") or []
    if not txn_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f44_morning_stakes_percentage"])

    total_stake: Dict[str, float] = {}
    morning_stake: Dict[str, float] = {}

    coerced_stake_warned = False

    for df in iter_csv_chunks(
        paths=txn_paths,
        usecols=[
            "Player_Profile_ID",
            "Transaction_Amount",
            "Transaction_Datetime",
            "Transaction_Type",
            "Transaction_Status",
        ],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        # timestamps
        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df[ts.notna()].copy()
        if df.empty:
            continue
        df["ts"] = ts[ts.notna()].values

        if start_datum is not None:
            m = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not m.any():
                continue
            df = df.loc[m].copy()

        # filter successful stakes
        typ = df["Transaction_Type"].fillna("").astype(str).str.upper()
        st  = df["Transaction_Status"].fillna("").astype(str).str.upper()
        df = df[(typ == "STAKE") & (st == "SUCCESSFUL")].copy()
        if df.empty:
            continue

        df["amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        df = df[df["amount"].notna()].copy()
        if df.empty:
            continue

        # enforce STAKE negative if someone encoded it positive
        to_flip = df["amount"] > 0
        if to_flip.any():
            df.loc[to_flip, "amount"] = -df.loc[to_flip, "amount"]
            if logger and (not coerced_stake_warned):
                logger.warning("⚠️ Coerced positive STAKE amounts to negative (one-time warning).")
            coerced_stake_warned = True

        df["hour"] = df["ts"].dt.hour.astype("int16")
        is_morning = (df["hour"] >= 8) & (df["hour"] < 16)

        # aggregate
        for pid, amt, morn in zip(df["Player_Profile_ID"].astype(str), df["amount"].astype(float), is_morning.to_numpy()):
            a = abs(amt)
            total_stake[pid] = total_stake.get(pid, 0.0) + a
            if morn:
                morning_stake[pid] = morning_stake.get(pid, 0.0) + a

    pids = sorted(total_stake.keys())
    records = []
    for pid in pids:
        tot = total_stake.get(pid, 0.0)
        val = (morning_stake.get(pid, 0.0) / tot) if tot > 0 else np.nan
        records.append({"Player_Profile_ID": pid, "f44_morning_stakes_percentage": float(val) if pd.notna(val) else np.nan})

    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "f44_morning_stakes_percentage"]
    )
    if logger:
        logger.info(f"✅ F44 klaar: {len(out):,} spelers")
    return out

# ------------------------------
# F45: evening stakes percentage
# ------------------------------

def f45_evening_stakes_percentage(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F45 (spec):
    Share (0..1) of amounts wagered (stakes) during the specified period that occur in:
      16:00:00 to 23:59:59  =>  16 <= hour < 24

    Uses WOK_Player_Account_Transaction:
      - Transaction_Type == "STAKE"
      - Transaction_Status == "SUCCESSFUL"
      - amount aggregated as abs(Transaction_Amount)

    Output:
      - f45_evening_stakes_percentage: float in [0,1], NaN if total stake == 0
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f45_evening_stakes_percentage")
        logger.info("▶ START F45: Evening stakes share (16:00-23:59)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    txn_paths = tables.get("WOK_Player_Account_Transaction") or []
    if not txn_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f45_evening_stakes_percentage"])

    total_stake: Dict[str, float] = {}
    evening_stake: Dict[str, float] = {}

    coerced_stake_warned = False

    for df in iter_csv_chunks(
        paths=txn_paths,
        usecols=[
            "Player_Profile_ID",
            "Transaction_Amount",
            "Transaction_Datetime",
            "Transaction_Type",
            "Transaction_Status",
        ],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df[ts.notna()].copy()
        if df.empty:
            continue
        df["ts"] = ts[ts.notna()].values

        if start_datum is not None:
            m = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not m.any():
                continue
            df = df.loc[m].copy()

        typ = df["Transaction_Type"].fillna("").astype(str).str.upper()
        st  = df["Transaction_Status"].fillna("").astype(str).str.upper()
        df = df[(typ == "STAKE") & (st == "SUCCESSFUL")].copy()
        if df.empty:
            continue

        df["amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        df = df[df["amount"].notna()].copy()
        if df.empty:
            continue

        to_flip = df["amount"] > 0
        if to_flip.any():
            df.loc[to_flip, "amount"] = -df.loc[to_flip, "amount"]
            if logger and (not coerced_stake_warned):
                logger.warning("⚠️ Coerced positive STAKE amounts to negative (one-time warning).")
            coerced_stake_warned = True

        df["hour"] = df["ts"].dt.hour.astype("int16")
        is_evening = (df["hour"] >= 16) & (df["hour"] < 24)

        for pid, amt, eve in zip(df["Player_Profile_ID"].astype(str), df["amount"].astype(float), is_evening.to_numpy()):
            a = abs(amt)
            total_stake[pid] = total_stake.get(pid, 0.0) + a
            if eve:
                evening_stake[pid] = evening_stake.get(pid, 0.0) + a

    pids = sorted(total_stake.keys())
    records = []
    for pid in pids:
        tot = total_stake.get(pid, 0.0)
        val = (evening_stake.get(pid, 0.0) / tot) if tot > 0 else np.nan
        records.append({"Player_Profile_ID": pid, "f45_evening_stakes_percentage": float(val) if pd.notna(val) else np.nan})

    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "f45_evening_stakes_percentage"]
    )
    if logger:
        logger.info(f"✅ F45 klaar: {len(out):,} spelers")
    return out

# ------------------------------
# F46: Median seconds between bet placed and bet resolved
# ------------------------------

def f46_median_seconds_bet_placed_to_resolved(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F46.
    Median of the time differences (seconds) between a bet being placed and that bet being resolved
    during the specified period. Positive real number. N/A if no bets are placed.

    Practical implementation for CDB/WOK (organisatie-relationeel):
    - Source: WOK_Bet (één rij per bet, met de eindstatus in Bet_Status).
    - placed_ts   = Bet_Start_Datetime
    - resolved_ts = WOK_Bet.Extraction_Date van diezelfde rij, mits de status afgewikkeld is
        (SETTLED/RESOLVED/CLOSED/CANCELLED/VOID/WON/LOST); BET_PLACED = open → geen resolved-tijd.
        Dit is 1-op-1 de originele root-berekening: die kende geen Bet_Resolved_Datetime-kolom in dit
        formaat en viel terug op Extraction_Date. created_at wordt NIET gebruikt (net als in de root).
    - Player_Profile_ID komt uit WOK_Bet_Transaction (WOK_Bet heeft geen speler-kolom), join op pk_id.
    - Filters on event timestamps in [start, end) (no +1 day logic).
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f46_median_seconds_bet_placed_to_resolved")
        logger.info("▶ START F46: Median seconds bet placed → resolved")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    if not x_tijdspad:
        if logger:
            logger.warning("⚠️ F46 requires x_tijdspad (period end needed). Returning empty.")
        return pd.DataFrame(columns=["Player_Profile_ID", "f46_median_seconds_bet_placed_to_resolved"])

    start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
    eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])

    bet_paths = tables.get("WOK_Bet")
    if not bet_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f46_median_seconds_bet_placed_to_resolved"])

    # organisatie-relationeel: WOK_Bet is één rij per bet met de EINDSTATUS (BET_PLACED/WON/LOST/CANCELLED/
    # SETTLED). De afwikkeltijd staat op die rij zélf — Extraction_Date — exact de fallback die de
    # originele root-code gebruikte toen er geen Bet_Resolved_Datetime-kolom was. created_at gebruiken
    # we NIET (deed de root ook niet). placed = Bet_Start_Datetime. De speler komt uit
    # WOK_Bet_Transaction (WOK_Bet heeft geen speler-kolom), join op pk_id → wok_bet_pk_id.
    _players_by_bet = build_bet_players_map(tables.get("WOK_Bet_Transaction"), chunksize=chunksize)

    # "resolved"-statussen: zelfde set als de root (SETTLED/RESOLVED/CLOSED/CANCELLED/VOID/WON/LOST).
    # BET_PLACED = open → nog geen resolved-tijd. Substring-match dekt de organisatie-prefix (BET_*).
    resolved_keywords = ("SETTLED", "RESOLVED", "CLOSED", "CANCELLED", "VOID", "WON", "LOST")

    # collected deltas per player
    deltas_per_player: Dict[str, List[float]] = {}

    # track whether a player placed *any* bet (for N/A semantics)
    placed_any_bet: Dict[str, bool] = {}

    for df in iter_csv_chunks(
        paths=bet_paths,
        usecols=["pk_id", "Bet_Start_Datetime", "Bet_Status", "Extraction_Date"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df is None or df.empty:
            continue

        placed_ts = pd.to_datetime(df["Bet_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        resolved_ts = pd.to_datetime(df["Extraction_Date"], errors="coerce", utc=True).dt.tz_localize(None)
        df["_placed_ts"] = placed_ts
        df["_resolved_ts"] = resolved_ts
        df = df[df["_placed_ts"].notna()].copy()
        if df.empty:
            continue

        # filter placed events in [start,end)
        mask_period = (df["_placed_ts"] >= start_datum) & (df["_placed_ts"] < eind_datum)
        if not mask_period.any():
            continue
        df = df.loc[mask_period].copy()

        idx_pk  = df.columns.get_loc("pk_id")
        idx_pts = df.columns.get_loc("_placed_ts")
        idx_rts = df.columns.get_loc("_resolved_ts")
        idx_st  = df.columns.get_loc("Bet_Status")

        for i in range(len(df)):
            pk = df.iat[i, idx_pk]
            players = _players_by_bet.get(pk, set())
            if not players:
                # can't attribute to a player -> skip (zoals de root)
                continue

            # mark that these players placed a bet in the window
            for pid in players:
                placed_any_bet[pid] = True

            status_raw = str(df.iat[i, idx_st]).upper().strip()
            if not any(k in status_raw for k in resolved_keywords):
                # bet nog niet afgewikkeld (BET_PLACED) → geen resolved-tijd
                continue

            placed_ts_i = df.iat[i, idx_pts]
            resolved_ts_i = df.iat[i, idx_rts]
            if pd.isna(resolved_ts_i) or pd.isna(placed_ts_i):
                continue
            try:
                delta = (resolved_ts_i - placed_ts_i).total_seconds()
            except Exception:
                continue
            if delta <= 0:
                # ignore pathological ordering / bad timestamps
                continue

            for pid in players:
                deltas_per_player.setdefault(pid, []).append(float(delta))

    # Build output: N/A if no bets placed
    all_players = set(placed_any_bet.keys()) | set(deltas_per_player.keys())
    records = []
    for pid in all_players:
        if not placed_any_bet.get(pid, False):
            # spec: N/A if no bets are placed
            val = np.nan
        else:
            ds = deltas_per_player.get(pid, [])
            val = float(np.median(ds)) if ds else np.nan  # N/A if no resolved after placed
        records.append({"Player_Profile_ID": pid, "f46_median_seconds_bet_placed_to_resolved": val})

    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "f46_median_seconds_bet_placed_to_resolved"]
    )

    if logger:
        logger.info(f"✅ F46 klaar: {len(out):,} spelers")
        if len(out) > 0:
            na = out["f46_median_seconds_bet_placed_to_resolved"].isna().sum()
            logger.info(f"   N/A: {na:,}")

    return out

# ------------------------------
# F47: Median seconds between session start and period end
# ------------------------------

def f47_median_seconds_session_start_to_period_end(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F47.
    Median of the time differences (seconds) between the start of a session of other games
    and the end of the specified period. Positive real number.
    N/A if no sessions of other games are played.

    Implementation:
    - Source: WOK_Game_Session
    - For each session with start_ts in [start,end):
        delta = (end_datum - start_ts).total_seconds()
    - Player_Profile_ID taken from column if present; else from Game_Transactions JSON.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f47_median_seconds_session_start_to_period_end")
        logger.info("▶ START F47: Median seconds session start → period end")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    if not x_tijdspad:
        if logger:
            logger.warning("⚠️ F47 requires x_tijdspad (period end needed). Returning empty.")
        return pd.DataFrame(columns=["Player_Profile_ID", "f47_median_seconds_session_start_to_period_end"])

    start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
    eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])

    session_paths = tables.get("WOK_Game_Session")
    if not session_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f47_median_seconds_session_start_to_period_end"])

    # organisatie-relationeel: transacties per sessie uit WOK_Game_Session_Transaction (join op pk_id).
    session_tx = build_session_tx_map(tables.get("WOK_Game_Session_Transaction"), chunksize=chunksize)

    deltas_per_player: Dict[str, List[float]] = {}
    played_any_session: Dict[str, bool] = {}

    for df in iter_csv_chunks(
        paths=session_paths,
        usecols=["pk_id", "Game_Session_Start_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df is None or df.empty:
            continue

        cols = df.columns.tolist()
        if "Game_Session_Start_Datetime" not in cols:
            continue

        ts = pd.to_datetime(df["Game_Session_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["_start_ts"] = ts
        df = df[df["_start_ts"].notna()].copy()
        if df.empty:
            continue

        mask = (df["_start_ts"] >= start_datum) & (df["_start_ts"] < eind_datum)
        if not mask.any():
            continue
        df = df.loc[mask].copy()

        idx_ts = df.columns.get_loc("_start_ts")
        idx_pk = df.columns.get_loc("pk_id")

        for i in range(len(df)):
            start_ts_i = df.iat[i, idx_ts]
            try:
                delta = (eind_datum - start_ts_i).total_seconds()
            except Exception:
                continue
            if delta <= 0:
                continue

            # players uit de relationele WOK_Game_Session_Transaction (join op pk_id)
            pids: List[str] = []
            pk = df.iat[i, idx_pk]
            seen = set()
            for pid, _tx in session_tx.get(pk, []):
                if pid is not None and not (isinstance(pid, float) and pd.isna(pid)):
                    sp = str(pid).strip()
                    if sp and sp not in seen:
                        seen.add(sp)
                        pids.append(sp)

            if not pids:
                continue

            for pid in pids:
                played_any_session[pid] = True
                deltas_per_player.setdefault(pid, []).append(float(delta))

    all_players = set(played_any_session.keys()) | set(deltas_per_player.keys())
    records = []
    for pid in all_players:
        if not played_any_session.get(pid, False):
            val = np.nan
        else:
            ds = deltas_per_player.get(pid, [])
            val = float(np.median(ds)) if ds else np.nan
        records.append({"Player_Profile_ID": pid, "f47_median_seconds_session_start_to_period_end": val})

    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "f47_median_seconds_session_start_to_period_end"]
    )

    if logger:
        logger.info(f"✅ F47 klaar: {len(out):,} spelers")
        if len(out) > 0:
            na = out["f47_median_seconds_session_start_to_period_end"].isna().sum()
            logger.info(f"   N/A: {na:,}")

    return out

# ------------------------------
# F48: Percentage of bets with cash-out (proxy: BET_UPDATED)
# ------------------------------

def f48_percentage_bets_with_cashout(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F48: Percentage of bets with cash-out.

    Definitie (CDB datamodel, via de cash-out-TRANSACTIE — 1-op-1 met de root):
    - Noemer: aantal unieke Bet_ID's geplaatst in de periode (Bet_Start_Datetime in [start, end)).
    - Teller: daarvan de bets met >= 1 cash-out. Een bet heeft een cash-out als één van zijn
      transacties (WOK_Bet_Transaction.transactions_id) voorkomt in WOK_Player_Account_Transaction
      met Transaction_Type == "CASH_OUT" en Transaction_Status == "SUCCESSFUL".

    Koppeling (relationeel): WOK_Bet_Transaction.transactions_id <-> WOK_Player_Account_Transaction.transaction_id
    (in het oude format: Bet_Transactions.Transaction_ID <-> WOK_Player_Account_Transaction.Transaction_ID).

    Output:
    - f48_percentage_bets_with_cashout: float in [0,1], of NaN als er geen bets zijn.

    Tijdsfilter: geen '+1 day' logica; strikt [start, end) op Bet_Start_Datetime. De cash-out-
    transactie zelf wordt NIET op de periode gefilterd (kan ná het bet-venster vallen).

    Benodigde input:
    - WOK_Bet: pk_id, Bet_ID, Bet_Start_Datetime
    - WOK_Bet_Transaction: wok_bet_pk_id, player_profile_id, transactions_id
    - WOK_Player_Account_Transaction: Transaction_ID, Transaction_Type, Transaction_Status
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f48_percentage_bets_with_cashout")
        logger.info("▶ START F48: % bets with cash-out (via CASH_OUT-transactie)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # ---- tijdspad ----
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    bet_paths = tables.get("WOK_Bet")
    if not bet_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f48_percentage_bets_with_cashout"])

    # ---- Stap 1: Transaction_ID's van geslaagde CASH_OUT-transacties ----
    cashout_txids: set = set()
    for df in iter_csv_chunks(
        paths=tables.get("WOK_Player_Account_Transaction") or [],
        usecols=["Transaction_ID", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize, verbose=verbose,
    ):
        if df is None or df.empty:
            continue
        m = (df["Transaction_Type"].astype(str).str.upper() == "CASH_OUT") & \
            (df["Transaction_Status"].astype(str).str.upper() == "SUCCESSFUL")
        for txid in df.loc[m, "Transaction_ID"].tolist():
            if txid is not None and not (isinstance(txid, float) and pd.isna(txid)):
                cashout_txids.add(str(txid))

    # ---- Stap 2: spelers + txids per bet via WOK_Bet_Transaction-join (relationeel i.p.v. JSON) ----
    # We tellen unieke Bet_ID's per speler: total_bets & cashout_bets.
    total_bets_per_pid: Dict[str, set] = {}
    cashout_bets_per_pid: Dict[str, set] = {}

    bet_tx = build_bet_tx_map(tables.get("WOK_Bet_Transaction"), chunksize=chunksize)

    for df in iter_csv_chunks(
        paths=bet_paths,
        usecols=["pk_id", "Bet_ID", "Bet_Start_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        # Timestamp parse + filter
        ts = pd.to_datetime(df["Bet_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        # basic hygiene
        df = df[df["Bet_ID"].notna()].copy()
        if df.empty:
            continue

        idx_bet_id = df.columns.get_loc("Bet_ID")
        idx_pk     = df.columns.get_loc("pk_id")

        for i in range(len(df)):
            bet_id = df.iat[i, idx_bet_id]
            if bet_id is None or (isinstance(bet_id, float) and np.isnan(bet_id)):
                continue
            bet_id = str(bet_id)

            pk = df.iat[i, idx_pk]

            # spelers + transactie-ids per bet via WOK_Bet_Transaction-join (relationeel i.p.v. JSON)
            pids: set = set()
            txids: set = set()
            for pid, txid in bet_tx.get(pk, []):
                if pid is not None and not (isinstance(pid, float) and pd.isna(pid)):
                    pids.add(str(pid))
                if txid is not None and not (isinstance(txid, float) and pd.isna(txid)):
                    txids.add(str(txid))

            if not pids:
                continue

            has_cashout = len(txids & cashout_txids) > 0
            for pid in pids:
                total_bets_per_pid.setdefault(pid, set()).add(bet_id)
                if has_cashout:
                    cashout_bets_per_pid.setdefault(pid, set()).add(bet_id)

    # Build per-player output
    all_pids = sorted(total_bets_per_pid.keys())
    records = []
    for pid in all_pids:
        total = len(total_bets_per_pid.get(pid, set()))
        if total == 0:
            val = np.nan
        else:
            cash = len(cashout_bets_per_pid.get(pid, set()))
            val = cash / total
        records.append({"Player_Profile_ID": pid, "f48_percentage_bets_with_cashout": val})

    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "f48_percentage_bets_with_cashout"]
    )

    if logger:
        logger.info(f"✅ F48 klaar: {len(out):,} spelers")
        if len(out) > 0:
            n_na = out["f48_percentage_bets_with_cashout"].isna().sum()
            logger.info(f"   N/A (no bets): {n_na:,}")
            valid = out["f48_percentage_bets_with_cashout"].dropna()
            if len(valid) > 0:
                logger.info(f"   Gemiddelde: {valid.mean():.4f}")

    return out

# ------------------------------
# F49: Percentage of live bets
# ------------------------------

def f49_percentage_live_bets(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F49: Percentage of live bets

    Spec:
      Percentage of live bets = (# live bets placed) / (total # bets placed) during reference period.
      Output is a real number in [0, 1], N/A (NaN) if no bet is registered.

    Implementation (CDB / WOK):
      - Table: WOK_Bet
      - Denominator: count of bets in-period (Bet_Start_Datetime in [start, end))
      - Numerator: count of those bets where Bet_Parts contains Part_Live == True
        (we treat a bet as "live" if ANY part is live)
      - If Bet_Parts has no parseable Part_Live anywhere -> we treat that bet as NOT live
        (conservative; avoids inflating % due to missing flags)

    Notes:
      - No '+1 day' logic. Filter is [start, end).
      - This is per player (Player_Profile_ID). If your schema uses a different player id column
        in WOK_Bet, adjust `pid_colname`.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f49_percentage_live_bets")
        logger.info("▶ START F49: Percentage live bets")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    if not x_tijdspad:
        if logger:
            logger.warning("⚠️ F49 requires x_tijdspad (reference period). Returning empty.")
        return pd.DataFrame(columns=["Player_Profile_ID", "f49_percentage_live_bets"])

    start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
    eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])

    bet_paths = tables.get("WOK_Bet")
    if not bet_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f49_percentage_live_bets"])

    # counts per player
    total_bets: Dict[str, int] = {}
    live_bets: Dict[str, int] = {}

    # Player_Profile_ID is NOT a top-level column in WOK_Bet.
    # organisatie-relationeel: speler via WOK_Bet_Transaction, live-flags via WOK_Bet_Parts (join op pk_id).
    _bet_tx_map = build_bet_tx_map(tables.get("WOK_Bet_Transaction"), chunksize=chunksize)
    _bet_parts_map = build_bet_parts_map(tables.get("WOK_Bet_Parts"), chunksize=chunksize)

    for df in iter_csv_chunks(
        paths=bet_paths,
        usecols=["pk_id", "Bet_Start_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        ts = pd.to_datetime(df["Bet_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()]
        if df.empty:
            continue

        mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
        if not mask.any():
            continue
        df = df.loc[mask].copy()

        idx_pk = df.columns.get_loc("pk_id")

        for i in range(len(df)):
            pk = df.iat[i, idx_pk]
            pids = set()
            for p, _t in _bet_tx_map.get(pk, []):
                if p is not None and not (isinstance(p, float) and pd.isna(p)):
                    pids.add(str(p))
            if not pids:
                continue

            any_live = False
            for part_obj in _bet_parts_map.get(pk, []):
                flag = part_obj.get("part_live")
                if isinstance(flag, str):
                    is_live = flag.strip().lower() in ("true", "1", "yes", "ja")
                else:
                    is_live = bool(flag) and not (isinstance(flag, float) and pd.isna(flag))
                if is_live:
                    any_live = True
                    break

            for pid in pids:
                total_bets[pid] = total_bets.get(pid, 0) + 1
                if any_live:
                    live_bets[pid] = live_bets.get(pid, 0) + 1

    # build output: include players with bets but 0 live
    records = []
    for pid, denom in total_bets.items():
        if denom <= 0:
            pct = np.nan
        else:
            pct = float(live_bets.get(pid, 0) / denom)
        records.append({"Player_Profile_ID": pid, "f49_percentage_live_bets": pct})

    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "f49_percentage_live_bets"]
    )

    if logger:
        logger.info(f"✅ F49 klaar: {len(out):,} spelers met ≥1 bet in periode")
        if len(out) > 0:
            logger.info(f"   Gemiddeld % live bets: {out['f49_percentage_live_bets'].mean():.4f}")

    return out

# ------------------------------
# F50: Single Bet % - Percentage of single bets vs combination bets
# ------------------------------

def f50_single_bet_percentage(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F50: Single Bet % (a.k.a. "simple bets")

    ✅ Definitie (robust / regulator-aligned):
      - "Simple bet" = bet met precies 1 selectie/onderdeel (n_parts == 1)
      - "Combination bet" = bet met >1 onderdelen (n_parts > 1)

    We leiden dit af uit Bet_Parts (JSON), NIET uit Bet_Type (operator-afhankelijk).

    Input:
      - WOK_Bet: Bet_Start_Datetime, Bet_Transactions, Bet_Parts

    Output:
      - f50_single_bet_percentage: float in [0,1], NaN als er geen bets zijn

    Notes:
      - Filter is [start, end) (geen +1 dag)
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f50_single_bet_percentage")
        logger.info("▶ START F50: Single Bet % (Bet_Parts-based)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # ---- tijdspad ----
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    bet_paths = tables.get("WOK_Bet")
    if not bet_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f50_single_bet_percentage"])

    # {pid: {"simple": int, "total": int}}
    counts: Dict[str, Dict[str, int]] = {}

    def _count_parts(bet_parts_cell) -> int:
        """
        Best-effort teller voor aantal bet-onderdelen.
        Ondersteunt o.a.:
          - {"Part": [ {...}, {...} ]}
          - [{"Part_ID":...}, ...]  (directe lijst)
          - {"Part_ID": "..."}      (enkele dict)
        """
        obj = _safe_load_json_relaxed(bet_parts_cell)
        if obj is None:
            return 0

        if isinstance(obj, dict):
            parts = obj.get("Part")
            if isinstance(parts, list):
                return sum(1 for x in parts if isinstance(x, dict))
            if isinstance(parts, dict):
                return 1
            # fallback: single-part dict
            if any(k in obj for k in ("Part_ID", "PartId", "Selection_ID", "SelectionId")):
                return 1
            return 0

        if isinstance(obj, list):
            return sum(1 for x in obj if isinstance(x, dict))

        return 0

    # organisatie-relationeel: spelers per bet (WOK_Bet_Transaction) + parts per bet (WOK_Bet_Parts), join op pk_id.
    _bet_tx_map = build_bet_tx_map(tables.get("WOK_Bet_Transaction"), chunksize=chunksize)
    _bet_parts_map = build_bet_parts_map(tables.get("WOK_Bet_Parts"), chunksize=chunksize)

    for df in iter_csv_chunks(
        paths=bet_paths,
        usecols=["pk_id", "Bet_Start_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        # time filter
        if start_datum is not None:
            ts = pd.to_datetime(df["Bet_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask = (ts >= start_datum) & (ts < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        if df.empty:
            continue

        idx_pk = df.columns.get_loc("pk_id")

        for i in range(len(df)):
            pk = df.iat[i, idx_pk]

            # player id (meestal exact 1) via WOK_Bet_Transaction-join
            pids = [str(p) for p, _t in _bet_tx_map.get(pk, [])
                    if p is not None and not (isinstance(p, float) and pd.isna(p))]
            if not pids:
                continue
            pid = str(pids[0])

            # aantal parts via WOK_Bet_Parts-join
            n_parts = len(_bet_parts_map.get(pk, []))
            if n_parts <= 0:
                n_parts = 0

            if pid not in counts:
                counts[pid] = {"simple": 0, "total": 0}

            counts[pid]["total"] += 1
            if n_parts == 1:
                counts[pid]["simple"] += 1

    records = []
    for pid, c in counts.items():
        total = c["total"]
        simple = c["simple"]
        val = (simple / total) if total > 0 else np.nan
        records.append({"Player_Profile_ID": pid, "f50_single_bet_percentage": val})

    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "f50_single_bet_percentage"]
    )

    if logger:
        logger.info(f"✅ F50 klaar: {len(out):,} spelers")
        if len(out) > 0:
            logger.info(f"   Gemiddelde single-bet fractie: {out['f50_single_bet_percentage'].mean(skipna=True):.4f}")

    return out

# ------------------------------
# F51: Median seconds loss to next bet (entire period)
# ------------------------------

def f51_median_seconds_loss_to_next_bet(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F51: Median seconds from a losing bet to the next bet placed, per player.

    Uses WOK_Player_Account_Transaction exclusively:
      - STAKE       → bet placed
      - WINNING     → preceding STAKE was won (not a loss)
      - VOID_BET    → preceding STAKE was voided (not a loss)
      - CASH_OUT    → not considered a loss

    For each player, consecutive STAKE pairs are examined. If no WINNING or
    VOID_BET occurred between STAKE[i] and STAKE[i+1], STAKE[i] is treated as
    a lost bet. The delta = STAKE[i+1] - STAKE[i] in seconds.
    F51 = median of all such deltas. NaN if no losing bets.

    x_tijdspad is applied to filter STAKE events (the period of activity).
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f51_median_seconds_loss_to_next_bet")
        logger.info("▶ START F51: Median seconds from losing STAKE to next STAKE (via WOK_Player_Account_Transaction)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]}")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f51_median_seconds_loss_to_next_bet"])

    NON_LOSS_TYPES = {"WINNING", "VOID_BET", "CASH_OUT"}

    # Per player: list of (datetime, transaction_type) sorted by time
    player_events: Dict[str, List] = {}

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        df = df[df["Player_Profile_ID"].notna() & df["Transaction_Type"].notna()]
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df.loc[ts.notna()].copy()
        ts = ts.loc[df.index]

        if start_datum is not None:
            mask = (ts >= start_datum) & (ts < eind_datum)
            df = df.loc[mask]
            ts = ts.loc[mask]

        if df.empty:
            continue

        for pid, t, typ in zip(df["Player_Profile_ID"], ts, df["Transaction_Type"]):
            player_events.setdefault(str(pid), []).append((t, str(typ).upper()))

    # Compute deltas per player
    records = []
    for pid, events in player_events.items():
        events.sort(key=lambda x: x[0])

        stakes = [(t, i) for i, (t, typ) in enumerate(events) if typ == "STAKE"]
        if len(stakes) < 2:
            records.append({"Player_Profile_ID": pid, "f51_median_seconds_loss_to_next_bet": np.nan})
            continue

        deltas = []
        for k in range(len(stakes) - 1):
            t_stake, idx_stake = stakes[k]
            t_next,  idx_next  = stakes[k + 1]

            # Check if any non-loss event occurred between the two STAKEs
            between = [typ for t, typ in events[idx_stake + 1:idx_next] if typ in NON_LOSS_TYPES]
            if not between:
                deltas.append((t_next - t_stake).total_seconds())

        med = float(np.median(deltas)) if deltas else np.nan
        records.append({"Player_Profile_ID": pid, "f51_median_seconds_loss_to_next_bet": med})

    out = pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=["Player_Profile_ID", "f51_median_seconds_loss_to_next_bet"]
    )

    if logger:
        logger.info(f"✅ F51 klaar: {len(out):,} spelers")
        if len(out) > 0:
            na = out["f51_median_seconds_loss_to_next_bet"].isna().sum()
            logger.info(f"   N/A (geen verloren bets): {na:,}")

    return out


# ------------------------------
# F52: Big-win wager increase count
# ------------------------------

def f52_big_win_wager_increase_count(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F52 (SPEC):
    Number of times a player increases their wagers after receiving a large (major) win.

    We implement the SPEC as:
    - For each player, identify the MAJOR WIN = the single largest WINNING amount (EUR) in the period.
      If a player has no WINNING at all -> N/A.
    - Compare average of the 10 stake amounts AFTER that major win vs average of the 10 stake amounts BEFORE it.
      If fewer than 10 exist on either side, use as many as possible but require at least 5 on BOTH sides.
      If <5 on either side -> the major win does NOT count as an increase (-> 0).
    - Count 1 if post_avg >= 2 * pre_avg else 0.

    Data assumptions:
    - Uses WOK_Player_Account_Transaction with Transaction_Datetime sorted (you said we can assume sorting now).
    - Uses Transaction_Type in {"STAKE","WINNING"} and Transaction_Status == "SUCCESSFUL".
    - "Amounts played" interpreted as absolute stake amounts (abs(Transaction_Amount) for STAKE).

    Notes:
    - No '+1 day' logic. Time filter is [start, end) if x_tijdspad is provided.
    - We coerce positive STAKE amounts to negative (one-time warning in logger) per your convention.
    """

    if log_path:
        logger = _setup_feature_logger(log_path, "f52_big_win_wager_increase_count")
        logger.info("▶ START F52: Big-win wager increase count (0/1)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # ---- tijdspad ----
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    tx_paths = tables.get("WOK_Player_Account_Transaction")
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f52_big_win_wager_increase_count"])

    # ---------- PASS 1: find per-player major win (max WINNING amount) ----------
    # store: pid -> (max_win_amount, max_win_ts)
    major_win_amt: Dict[str, float] = {}
    major_win_ts: Dict[str, pd.Timestamp] = {}

    coerced_stake_warned = False

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=[
            "Player_Profile_ID",
            "Transaction_Datetime",
            "Transaction_Amount",
            "Transaction_Type",
            "Transaction_Status",
        ],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        df = df[df["Transaction_Status"] == "SUCCESSFUL"]
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()]
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        df["amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        df = df[df["amount"].notna()]
        if df.empty:
            continue

        # enforce STAKE negative (for consistency; pass1 doesn't need it but keep consistent)
        is_stake = df["Transaction_Type"].astype(str).eq("STAKE")
        to_flip = is_stake & (df["amount"] > 0)
        if to_flip.any():
            df.loc[to_flip, "amount"] = -df.loc[to_flip, "amount"]
            if (not coerced_stake_warned) and logger:
                logger.warning("⚠️ Coerced positive STAKE amounts to negative (one-time warning).")
            coerced_stake_warned = True

        # only WINNING rows matter in pass1
        dfw = df[df["Transaction_Type"].astype(str).eq("WINNING")]
        if dfw.empty:
            continue

        pid_col = dfw.columns.get_loc("Player_Profile_ID")
        ts_col  = dfw.columns.get_loc("ts")
        amt_col = dfw.columns.get_loc("amount")

        for i in range(len(dfw)):
            pid = str(dfw.iat[i, pid_col])
            amt = float(dfw.iat[i, amt_col])
            t   = dfw.iat[i, ts_col]
            # WINNING should be positive; still use abs to be robust
            win_amt = abs(amt)

            prev = major_win_amt.get(pid)
            if prev is None or win_amt > prev:
                major_win_amt[pid] = win_amt
                major_win_ts[pid] = t
            # tie: keep earliest (so "after" window has more chance)
            elif prev is not None and win_amt == prev:
                if pid in major_win_ts and t < major_win_ts[pid]:
                    major_win_ts[pid] = t

    if not major_win_amt:
        # no wins in entire dataset
        return pd.DataFrame(columns=["Player_Profile_ID", "f52_big_win_wager_increase_count"])

    # ---------- PASS 2: collect 10 stakes before and 10 after major win ----------
    # per player buffers
    pre_stakes: Dict[str, List[float]] = {}     # rolling last 10 stakes seen so far (while ts <= win_ts)
    post_stakes: Dict[str, List[float]] = {}    # first up to 10 stakes after win_ts
    finished: Dict[str, bool] = {}              # once post has 10, we can ignore further stakes for that pid

    # We'll stream again; since data are sorted by time, we can do:
    # - while ts <= win_ts: keep rolling window of last 10 stakes
    # - when ts > win_ts: start filling post stakes until 10
    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=[
            "Player_Profile_ID",
            "Transaction_Datetime",
            "Transaction_Amount",
            "Transaction_Type",
            "Transaction_Status",
        ],
        chunksize=chunksize,
        verbose=verbose,
    ):
        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        df = df[df["Transaction_Status"] == "SUCCESSFUL"]
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()]
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        df["amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        df = df[df["amount"].notna()]
        if df.empty:
            continue

        is_stake = df["Transaction_Type"].astype(str).eq("STAKE")
        to_flip = is_stake & (df["amount"] > 0)
        if to_flip.any():
            df.loc[to_flip, "amount"] = -df.loc[to_flip, "amount"]
            # warning already handled in pass1; no need again

        # stakes only
        dfs = df[df["Transaction_Type"].astype(str).eq("STAKE")]
        if dfs.empty:
            continue

        pid_col = dfs.columns.get_loc("Player_Profile_ID")
        ts_col  = dfs.columns.get_loc("ts")
        amt_col = dfs.columns.get_loc("amount")

        for i in range(len(dfs)):
            pid = str(dfs.iat[i, pid_col])
            if pid not in major_win_ts:
                continue
            if finished.get(pid, False):
                continue

            t = dfs.iat[i, ts_col]
            stake_amt = abs(float(dfs.iat[i, amt_col]))  # amounts played

            win_t = major_win_ts[pid]

            if t <= win_t:
                # rolling last 10
                lst = pre_stakes.setdefault(pid, [])
                lst.append(stake_amt)
                if len(lst) > 10:
                    lst.pop(0)
            else:
                lst = post_stakes.setdefault(pid, [])
                if len(lst) < 10:
                    lst.append(stake_amt)
                    if len(lst) >= 10:
                        finished[pid] = True

    # ---------- Compute result per player ----------
    records = []
    for pid in major_win_ts.keys():
        pre = pre_stakes.get(pid, [])
        post = post_stakes.get(pid, [])

        # need at least 5 on both sides
        if len(pre) < 5 or len(post) < 5:
            val = 0.0  # spec says: if fewer than five, largest win does not count as an increase
        else:
            pre_avg = float(np.mean(pre[-10:]))
            post_avg = float(np.mean(post[:10]))
            val = 1.0 if post_avg >= 2.0 * pre_avg else 0.0

        records.append({"Player_Profile_ID": pid, "f52_big_win_wager_increase_count": val})

    out = pd.DataFrame.from_records(records)

    # players without any WINNING should be N/A; we didn't include them at all.
    # if you want them present with NaN, you'd need an anchor set (e.g., F1 list). Keeping spec here.

    if logger:
        logger.info(f"✅ F52 klaar: {len(out):,} spelers met >=1 WINNING (others are N/A by omission).")
        if len(out) > 0:
            logger.info(f"   share==1: {(out['f52_big_win_wager_increase_count'] == 1.0).mean():.3f}")

    return out

# ------------------------------
# F53: Absolute gradient of amounts wagered around median date (index-median, not time-median)
# ------------------------------

def f53_abs_gradient_wagered_around_median_date(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F53: Absolute value of the gradient of amounts wagered relative to the median date.

    Spec (DGOJ):
      Absolute value of the difference between the sum of amounts wagered in the two
      sub-periods defined by the median date, divided by the number of interactions
      throughout the specified period.

    Our concrete interpretation (streaming-safe, deterministic):
      - "Amounts wagered" = abs(Transaction_Amount) for successful STAKE transactions
      - "Interactions"    = number of successful STAKE transactions (same set as wagered)
      - "Median date"     = the median interaction POSITION per player (not an interpolated timestamp):
            median_idx = (N - 1) // 2   (0-based over the player's chronologically processed STAKEs)
        Sub-periods:
          * before = stakes with index <= median_idx
          * after  = stakes with index  > median_idx

      F53 = abs(sum_after - sum_before) / N

    Notes:
      - No '+1 day' logic. Filter is [start, end).
      - Assumes WOK_Player_Account_Transaction is sorted by Transaction_Datetime (globally).
      - Enforces STAKE amounts as negative if encountered as positive (one-time warning).
      - Returns NaN if a player has no STAKE interactions in the period.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f53_abs_gradient_wagered_around_median_date")
        logger.info("▶ START F53: abs gradient wagered around median (index-median)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    # ---- tijdspad ----
    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    tx_paths = tables.get("WOK_Player_Account_Transaction") or []
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f53_abs_gradient_wagered_around_median_date"])

    # ============================
    # PASS 1: count N stakes per pid
    # ============================
    stake_counts: Dict[str, int] = {}
    coerced_stake_warned = False

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Amount", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        # successful only
        df = df[df["Transaction_Status"].astype(str).str.upper().eq("SUCCESSFUL")]
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        # keep only STAKE
        df["Transaction_Type"] = df["Transaction_Type"].astype(str)
        df = df[df["Transaction_Type"].str.upper().eq("STAKE")].copy()
        if df.empty:
            continue

        df["amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        df = df[df["amount"].notna()].copy()
        if df.empty:
            continue

        # enforce STAKE negative (project convention)
        to_flip = df["amount"] > 0
        if to_flip.any():
            df.loc[to_flip, "amount"] = -df.loc[to_flip, "amount"]
            if (not coerced_stake_warned) and logger:
                logger.warning("⚠️ Coerced positive STAKE amounts to negative (one-time warning).")
            coerced_stake_warned = True

        # count per pid
        vc = df["Player_Profile_ID"].astype(str).value_counts()
        for pid, n in vc.items():
            stake_counts[pid] = stake_counts.get(pid, 0) + int(n)

    if not stake_counts:
        return pd.DataFrame(columns=["Player_Profile_ID", "f53_abs_gradient_wagered_around_median_date"])

    # median index per player (0-based, inclusive in "before")
    median_idx: Dict[str, int] = {pid: (n - 1) // 2 for pid, n in stake_counts.items() if n > 0}

    # ============================
    # PASS 2: accumulate sums before/after by median position
    # ============================
    seen_idx: Dict[str, int] = {}      # pid -> next stake index to assign (0,1,2,...)
    sum_before: Dict[str, float] = {}  # pid -> sum abs stakes <= median_idx
    sum_after: Dict[str, float] = {}   # pid -> sum abs stakes >  median_idx

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Amount", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        df = df[df["Transaction_Status"].astype(str).str.upper().eq("SUCCESSFUL")]
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        df["Transaction_Type"] = df["Transaction_Type"].astype(str)
        df = df[df["Transaction_Type"].str.upper().eq("STAKE")].copy()
        if df.empty:
            continue

        df["amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        df = df[df["amount"].notna()].copy()
        if df.empty:
            continue

        # enforce STAKE negative consistently
        to_flip = df["amount"] > 0
        if to_flip.any():
            df.loc[to_flip, "amount"] = -df.loc[to_flip, "amount"]

        # IMPORTANT: we assume the full input is already sorted by Transaction_Datetime,
        # so within chunk ordering is fine; no per-player sort needed.

        pid_col = df.columns.get_loc("Player_Profile_ID")
        amt_col = df.columns.get_loc("amount")

        for i in range(len(df)):
            pid = str(df.iat[i, pid_col])
            if pid not in median_idx:
                continue  # player not in pass1 (should be rare)

            # current stake index for this pid
            j = seen_idx.get(pid, 0)
            seen_idx[pid] = j + 1

            wager = abs(float(df.iat[i, amt_col]))
            if j <= median_idx[pid]:
                sum_before[pid] = sum_before.get(pid, 0.0) + wager
            else:
                sum_after[pid] = sum_after.get(pid, 0.0) + wager

    # ============================
    # Build output
    # ============================
    records = []
    for pid, n in stake_counts.items():
        if n <= 0:
            val = np.nan
        else:
            b = float(sum_before.get(pid, 0.0))
            a = float(sum_after.get(pid, 0.0))
            val = abs(a - b) / float(n)
        records.append({"Player_Profile_ID": pid, "f53_abs_gradient_wagered_around_median_date": val})

    out = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ F53 klaar: {len(out):,} spelers")
        if len(out) > 0:
            logger.info(f"   mean={out['f53_abs_gradient_wagered_around_median_date'].mean():.4f}")

    return out

# ------------------------------
# F54: Percentage of days with interaction occurring after the median date, relative to total activity days (index-median)
# ------------------------------

def f54_post_median_active_days_percentage(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F54 (DGOJ): Percentage of days with interaction occurring after the median date,
    relative to total activity days.

    Spec interpretation (streaming-safe, deterministic):
      1) Consider the interaction stream for the player during the period.
         Here we use SUCCESSFUL STAKE transactions in WOK_Player_Account_Transaction
         as the interaction proxy (same basis as F53 wagering interactions).
      2) Define the median date via the median interaction POSITION per player:
           median_idx = (N - 1) // 2   over the player's chronologically processed STAKE interactions.
      3) Split interactions into two halves by that index:
           - first half: indices <= median_idx
           - second half: indices  > median_idx
      4) For each half, count UNIQUE calendar days (date part of Transaction_Datetime).
      5) Return: days_second_half / (days_first_half + days_second_half)
         -> real number in [0, 1]
      6) Return NaN if the player has no interactions (no stakes) in the period.

    Notes:
      - No '+1 day' logic. Filter is [start, end).
      - Assumes input is globally sorted by Transaction_Datetime.
      - Does NOT compute a median timestamp; uses median position (stable, streaming-friendly).
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f54_post_median_active_days_percentage")
        logger.info("▶ START F54: post-median active days share (index-median)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    tx_paths = tables.get("WOK_Player_Account_Transaction") or []
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f54_post_median_active_days_percentage"])

    
    # PASS 1: count N interactions (stakes) per pid
    stake_counts: Dict[str, int] = {}

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        df = df[df["Transaction_Status"].astype(str).str.upper().eq("SUCCESSFUL")]
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        df["Transaction_Type"] = df["Transaction_Type"].astype(str)
        df = df[df["Transaction_Type"].str.upper().eq("STAKE")].copy()
        if df.empty:
            continue

        vc = df["Player_Profile_ID"].astype(str).value_counts()
        for pid, n in vc.items():
            stake_counts[pid] = stake_counts.get(pid, 0) + int(n)

    if not stake_counts:
        return pd.DataFrame(columns=["Player_Profile_ID", "f54_post_median_active_days_percentage"])

    median_idx: Dict[str, int] = {pid: (n - 1) // 2 for pid, n in stake_counts.items() if n > 0}

    # PASS 2: track unique days in first vs second half (by index)
    seen_idx: Dict[str, int] = {}
    days_first: Dict[str, set] = {}
    days_second: Dict[str, set] = {}

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        df = df[df["Transaction_Status"].astype(str).str.upper().eq("SUCCESSFUL")]
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        df["Transaction_Type"] = df["Transaction_Type"].astype(str)
        df = df[df["Transaction_Type"].str.upper().eq("STAKE")].copy()
        if df.empty:
            continue

        # day bucket
        df["day"] = df["ts"].dt.date

        pid_col = df.columns.get_loc("Player_Profile_ID")
        day_col = df.columns.get_loc("day")

        for i in range(len(df)):
            pid = str(df.iat[i, pid_col])
            if pid not in median_idx:
                continue

            j = seen_idx.get(pid, 0)
            seen_idx[pid] = j + 1

            d = df.iat[i, day_col]
            if j <= median_idx[pid]:
                if pid not in days_first:
                    days_first[pid] = set()
                days_first[pid].add(d)
            else:
                if pid not in days_second:
                    days_second[pid] = set()
                days_second[pid].add(d)


    # Build output
    records = []
    for pid, n in stake_counts.items():
        if n <= 0:
            val = np.nan
        else:
            a = len(days_first.get(pid, set()))
            b = len(days_second.get(pid, set()))
            denom = a + b
            val = (b / denom) if denom > 0 else np.nan

        records.append({"Player_Profile_ID": pid, "f54_post_median_active_days_percentage": val})

    out = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ F54 klaar: {len(out):,} spelers")
        if len(out) > 0:
            valid = out["f54_post_median_active_days_percentage"].dropna()
            if len(valid) > 0:
                logger.info(f"   mean={valid.mean():.4f}")

    return out

# ------------------------------
# F55: Stake Variance Difference - abs(var_after - var_before) around median interaction
# ------------------------------

def f55_stake_variance_difference(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F55 (DGOJ):
    Absolute value of the difference between the variances of the amounts wagered
    before and after the median date.

    Streaming implementation:
    - Use SUCCESSFUL STAKE transactions from WOK_Player_Account_Transaction as "amounts wagered".
    - Define the split by the median interaction POSITION per player (sorted by Transaction_Datetime):
        median_idx = (N - 1) // 2, where N = number of stake interactions in period.
      First half: indices <= median_idx
      Second half: indices  > median_idx
    - Compute sample variance (ddof=1) per half (needs >=2 values per half).
    - Output = abs(var_second - var_first) (non-negative).
    - Output NaN if insufficient data (no stakes, or <2 stakes in either half).

    Notes:
    - No '+1 day' logic. Filter is [start, end).
    - Assumes files are globally sorted by Transaction_Datetime.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f55_stake_variance_difference")
        logger.info("▶ START F55: abs(var_after - var_before) using median-index split")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    tx_paths = tables.get("WOK_Player_Account_Transaction") or []
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f55_stake_variance_difference"])

    stake_counts: Dict[str, int] = {}

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        df = df[df["Transaction_Status"].astype(str).str.upper().eq("SUCCESSFUL")]
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        df["Transaction_Type"] = df["Transaction_Type"].astype(str)
        df = df[df["Transaction_Type"].str.upper().eq("STAKE")].copy()
        if df.empty:
            continue

        vc = df["Player_Profile_ID"].astype(str).value_counts()
        for pid, n in vc.items():
            stake_counts[pid] = stake_counts.get(pid, 0) + int(n)

    if not stake_counts:
        return pd.DataFrame(columns=["Player_Profile_ID", "f55_stake_variance_difference"])

    median_idx: Dict[str, int] = {pid: (n - 1) // 2 for pid, n in stake_counts.items() if n > 0}

    seen_idx: Dict[str, int] = {}

    n1: Dict[str, int] = {}
    s1: Dict[str, float] = {}
    ss1: Dict[str, float] = {}

    n2: Dict[str, int] = {}
    s2: Dict[str, float] = {}
    ss2: Dict[str, float] = {}

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=[
            "Player_Profile_ID",
            "Transaction_Datetime",
            "Transaction_Type",
            "Transaction_Status",
            "Transaction_Amount",
        ],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        df = df[df["Transaction_Status"].astype(str).str.upper().eq("SUCCESSFUL")]
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        df["Transaction_Type"] = df["Transaction_Type"].astype(str)
        df = df[df["Transaction_Type"].str.upper().eq("STAKE")].copy()
        if df.empty:
            continue

        amt = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        df["amt"] = amt
        df = df[df["amt"].notna()].copy()
        if df.empty:
            continue

        pid_col = df.columns.get_loc("Player_Profile_ID")
        amt_col = df.columns.get_loc("amt")

        for i in range(len(df)):
            pid = str(df.iat[i, pid_col])
            if pid not in median_idx:
                continue

            x = abs(float(df.iat[i, amt_col]))  # magnitude wagered

            j = seen_idx.get(pid, 0)
            seen_idx[pid] = j + 1

            if j <= median_idx[pid]:
                n1[pid] = n1.get(pid, 0) + 1
                s1[pid] = s1.get(pid, 0.0) + x
                ss1[pid] = ss1.get(pid, 0.0) + x * x
            else:
                n2[pid] = n2.get(pid, 0) + 1
                s2[pid] = s2.get(pid, 0.0) + x
                ss2[pid] = ss2.get(pid, 0.0) + x * x

    def _sample_var(n: int, s: float, ss: float) -> float:
        if n < 2:
            return np.nan
        return (ss - (s * s) / n) / (n - 1)

    records = []
    for pid, _N in stake_counts.items():
        v1 = _sample_var(n1.get(pid, 0), s1.get(pid, 0.0), ss1.get(pid, 0.0))
        v2 = _sample_var(n2.get(pid, 0), s2.get(pid, 0.0), ss2.get(pid, 0.0))

        val = np.nan if (np.isnan(v1) or np.isnan(v2)) else float(abs(v2 - v1))
        records.append({"Player_Profile_ID": pid, "f55_stake_variance_difference": val})

    out = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ F55 klaar: {len(out):,} spelers")
        valid = out["f55_stake_variance_difference"].dropna()
        if len(valid) > 0:
            logger.info(f"   mean={valid.mean():.4f}")

    return out

# ------------------------------
# F56: Stake CV Difference - abs(CV_after - CV_before) around median interaction
# ------------------------------

def f56_stake_cv_difference(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F56 (DGOJ-style):
    Difference in coefficient of variation (CV = std/mean) of wagered amounts
    before vs after the player's median interaction (position split).

    Streaming implementation (2-pass):
    - Use SUCCESSFUL STAKE transactions as "amounts wagered".
    - Assume transactions are globally sorted by Transaction_Datetime.
    - Per player: median_idx = (N - 1) // 2 where N = number of stake interactions in period.
      First half: indices <= median_idx
      Second half: indices  > median_idx
    - Compute sample std per half via sums/sumsquares (ddof=1), then CV=std/mean.
    - Output = abs(CV_second - CV_first) (non-negative).
    - NaN if insufficient data (needs >=2 stakes per half, and mean>0 in both halves).

    Notes:
    - No '+1 day' logic. Filter is [start, end).
    """

    if log_path:
        logger = _setup_feature_logger(log_path, "f56_stake_cv_difference")
        logger.info("▶ START F56: abs(CV_after - CV_before) using median-index split")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    tx_paths = tables.get("WOK_Player_Account_Transaction") or []
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f56_stake_cv_difference"])

    stake_counts: Dict[str, int] = {}

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime", "Transaction_Type", "Transaction_Status"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        df = df[df["Transaction_Status"].astype(str).str.upper().eq("SUCCESSFUL")]
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        df["Transaction_Type"] = df["Transaction_Type"].astype(str)
        df = df[df["Transaction_Type"].str.upper().eq("STAKE")].copy()
        if df.empty:
            continue

        vc = df["Player_Profile_ID"].astype(str).value_counts()
        for pid, n in vc.items():
            stake_counts[pid] = stake_counts.get(pid, 0) + int(n)

    if not stake_counts:
        return pd.DataFrame(columns=["Player_Profile_ID", "f56_stake_cv_difference"])

    median_idx: Dict[str, int] = {pid: (n - 1) // 2 for pid, n in stake_counts.items() if n > 0}

    seen_idx: Dict[str, int] = {}

    n1: Dict[str, int] = {}
    s1: Dict[str, float] = {}
    ss1: Dict[str, float] = {}

    n2: Dict[str, int] = {}
    s2: Dict[str, float] = {}
    ss2: Dict[str, float] = {}

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=[
            "Player_Profile_ID",
            "Transaction_Datetime",
            "Transaction_Type",
            "Transaction_Status",
            "Transaction_Amount",
        ],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        df = df[df["Transaction_Status"].astype(str).str.upper().eq("SUCCESSFUL")]
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        df["Transaction_Type"] = df["Transaction_Type"].astype(str)
        df = df[df["Transaction_Type"].str.upper().eq("STAKE")].copy()
        if df.empty:
            continue

        amt = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        df["amt"] = amt
        df = df[df["amt"].notna()].copy()
        if df.empty:
            continue

        pid_col = df.columns.get_loc("Player_Profile_ID")
        amt_col = df.columns.get_loc("amt")

        for i in range(len(df)):
            pid = str(df.iat[i, pid_col])
            if pid not in median_idx:
                continue

            x = abs(float(df.iat[i, amt_col]))
            j = seen_idx.get(pid, 0)
            seen_idx[pid] = j + 1

            if j <= median_idx[pid]:
                n1[pid] = n1.get(pid, 0) + 1
                s1[pid] = s1.get(pid, 0.0) + x
                ss1[pid] = ss1.get(pid, 0.0) + x * x
            else:
                n2[pid] = n2.get(pid, 0) + 1
                s2[pid] = s2.get(pid, 0.0) + x
                ss2[pid] = ss2.get(pid, 0.0) + x * x

    def _sample_var(n: int, s: float, ss: float) -> float:
        if n < 2:
            return np.nan
        return (ss - (s * s) / n) / (n - 1)

    records = []
    for pid, _N in stake_counts.items():
        n_first  = n1.get(pid, 0)
        n_second = n2.get(pid, 0)

        if n_first < 2 or n_second < 2:
            records.append({"Player_Profile_ID": pid, "f56_stake_cv_difference": np.nan})
            continue

        mean1 = s1.get(pid, 0.0) / n_first
        mean2 = s2.get(pid, 0.0) / n_second
        if mean1 <= 0 or mean2 <= 0:
            records.append({"Player_Profile_ID": pid, "f56_stake_cv_difference": np.nan})
            continue

        var1 = _sample_var(n_first, s1.get(pid, 0.0), ss1.get(pid, 0.0))
        var2 = _sample_var(n_second, s2.get(pid, 0.0), ss2.get(pid, 0.0))
        if np.isnan(var1) or np.isnan(var2) or var1 < 0 or var2 < 0:
            records.append({"Player_Profile_ID": pid, "f56_stake_cv_difference": np.nan})
            continue

        cv1 = (np.sqrt(var1) / mean1) if mean1 > 0 else np.nan
        cv2 = (np.sqrt(var2) / mean2) if mean2 > 0 else np.nan

        val = np.nan if (np.isnan(cv1) or np.isnan(cv2)) else float(abs(cv2 - cv1))
        records.append({"Player_Profile_ID": pid, "f56_stake_cv_difference": val})

    out = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ F56 klaar: {len(out):,} spelers")
        valid = out["f56_stake_cv_difference"].dropna()
        if len(valid) > 0:
            logger.info(f"   mean={valid.mean():.6f}")

    return out

# ------------------------------
# F57: Longest Daily Streak
# ------------------------------

def f57_longest_daily_streak(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F57:
    Number of days of the longest gambling streak (consecutive active days).
    Positive integer <= F1.

    Efficient streaming version:
    - Assumes WOK_Player_Account_Transaction is sorted by Transaction_Datetime.
    - Counts at most 1 interaction per player per calendar day (dedup within day).
    - Streak increments only when day is exactly previous_day + 1.
    - No '+1 day' logic. Filter is [start, end).
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f57_longest_daily_streak")
        logger.info("▶ START F57: Longest Daily Streak")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    tx_paths = tables.get("WOK_Player_Account_Transaction") or []
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f57_longest_daily_streak"])

    last_day: Dict[str, Any] = {}       # pid -> date (python date)
    current_streak: Dict[str, int] = {} # pid -> current consecutive streak
    max_streak: Dict[str, int] = {}     # pid -> max consecutive streak

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        df["day"] = df["ts"].dt.date

        pid_col = df.columns.get_loc("Player_Profile_ID")
        day_col = df.columns.get_loc("day")

        for i in range(len(df)):
            pid = str(df.iat[i, pid_col])
            day = df.iat[i, day_col]  # datetime.date

            prev = last_day.get(pid)
            if prev is not None and day == prev:
                continue

            if prev is None:
                current_streak[pid] = 1
                max_streak[pid] = 1
            else:
                delta = (day - prev).days
                if delta == 1:
                    current_streak[pid] = current_streak.get(pid, 1) + 1
                else:
                    current_streak[pid] = 1
                if current_streak[pid] > max_streak.get(pid, 1):
                    max_streak[pid] = current_streak[pid]

            last_day[pid] = day

    records = [{"Player_Profile_ID": pid, "f57_longest_daily_streak": int(ms)} for pid, ms in max_streak.items()]
    out = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ F57 klaar: {len(out):,} spelers")
        if len(out) > 0:
            logger.info(f"   Gemiddelde streak: {out['f57_longest_daily_streak'].mean():.2f} dagen")
            logger.info(f"   Langste streak: {out['f57_longest_daily_streak'].max()} dagen")

    return out

# ------------------------------
# F58: Longest Streak Ratio
# ------------------------------

def f58_longest_streak_ratio(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F58:
    Number of days of the longest gambling streak relative to the days of activity.
    Calculated as F57 / F1. Real number in [0, 1]. N/A if no activity (F1==0).

    Efficient streaming version:
    - Assumes WOK_Player_Account_Transaction is sorted by Transaction_Datetime (global).
    - Deduplicates within-day per player (multiple tx same day count as 1 active day).
    - No '+1 day' logic. Filter is [start, end).
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f58_longest_streak_ratio")
        logger.info("▶ START F58: Longest Streak Ratio (F57/F1)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    tx_paths = tables.get("WOK_Player_Account_Transaction") or []
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f58_longest_streak_ratio"])

    last_day: Dict[str, Any] = {}            # pid -> last seen date
    current_streak: Dict[str, int] = {}      # pid -> current streak
    max_streak: Dict[str, int] = {}          # pid -> max streak
    active_days_count: Dict[str, int] = {}   # pid -> F1 (distinct active days)

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        df["day"] = df["ts"].dt.date

        pid_col = df.columns.get_loc("Player_Profile_ID")
        day_col = df.columns.get_loc("day")

        for i in range(len(df)):
            pid = str(df.iat[i, pid_col])
            day = df.iat[i, day_col]

            prev = last_day.get(pid)
            if prev is not None and day == prev:
                continue  # same day already counted

            # update F1
            active_days_count[pid] = active_days_count.get(pid, 0) + 1

            # update streak
            if prev is None:
                current_streak[pid] = 1
                max_streak[pid] = 1
            else:
                delta = (day - prev).days
                if delta == 1:
                    current_streak[pid] = current_streak.get(pid, 1) + 1
                else:
                    current_streak[pid] = 1
                if current_streak[pid] > max_streak.get(pid, 1):
                    max_streak[pid] = current_streak[pid]

            last_day[pid] = day

    records = []
    for pid, f1 in active_days_count.items():
        if f1 <= 0:
            ratio = np.nan
        else:
            ratio = max_streak.get(pid, 1) / f1
        records.append({"Player_Profile_ID": pid, "f58_longest_streak_ratio": float(ratio)})

    out = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ F58 klaar: {len(out):,} spelers")
        if len(out) > 0:
            valid = out[out["f58_longest_streak_ratio"].notna()]
            if len(valid) > 0:
                logger.info(f"   Gemiddelde ratio: {valid['f58_longest_streak_ratio'].mean():.4f}")

    return out

# ------------------------------
# F59: Median Daily Time Off (hours), with IQR outlier removal on streak-boundary days
# ------------------------------

def f59_median_daily_time_off(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F59:
    Median daily time off, measured in hours, excluding atypical data identified using IQR.
    Atypical data rule (spec):
      - Compute Q1 and Q3 of ALL "daily time off" gaps for the player (IQR = Q3 - Q1)
      - Values associated with the FIRST or LAST day of a gambling streak that are outside
        (Q1 - 1.5*IQR, Q3 + 1.5*IQR) are excluded.
    Output: positive real < 24, N/A if there is no daily time off.

    Implementation notes:
      - Assumes WOK_Player_Account_Transaction is globally sorted by Transaction_Datetime.
      - "Daily time off" = gaps between consecutive interactions within the SAME calendar day,
        with 0 < gap_hours < 24.
      - No '+1 day' logic. Filter is [start, end).
      - Streaming, avoids storing all timestamps.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f59_median_daily_time_off")
        logger.info("▶ START F59: Median Daily Time Off (IQR on streak-boundary days)")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    txn_paths = tables.get("WOK_Player_Account_Transaction") or []
    if not txn_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f59_median_daily_time_off"])

    # pass-1 state
    last_ts: Dict[str, Any] = {}       # pid -> last timestamp seen
    last_day: Dict[str, Any] = {}      # pid -> last active day seen (date)
    pending_end_day: Dict[str, Any] = {}  # pid -> candidate "end of streak" day (date) to be finalized later

    gaps_all: Dict[str, List[float]] = {}          # pid -> all gap hours (for Q1/Q3)
    gaps_by_day: Dict[str, Dict[Any, List[float]]] = {}  # pid -> day -> list gap hours (only days with gaps)
    boundary_days_with_gaps: Dict[str, set] = {}   # pid -> set(day) but ONLY for days that have gaps

    for df in iter_csv_chunks(
        paths=txn_paths,
        usecols=["Player_Profile_ID", "Transaction_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        df["day"] = df["ts"].dt.date

        pid_col = df.columns.get_loc("Player_Profile_ID")
        ts_col  = df.columns.get_loc("ts")
        day_col = df.columns.get_loc("day")

        for i in range(len(df)):
            pid = str(df.iat[i, pid_col])
            t   = df.iat[i, ts_col]
            d   = df.iat[i, day_col]

            prev_day = last_day.get(pid)

            # streak boundary bookkeeping (based on active days)
            if prev_day is None:
                # first ever active day for this pid => streak start
                # mark as boundary only if we later see any gaps on that day
                # (we'll add it when we observe a gap and know the day)
                pass
            else:
                day_gap = (d - prev_day).days
                if day_gap > 1:
                    # prev_day is end of a streak, current d is start of new streak
                    pending_end_day[pid] = prev_day
                    # start day 'd' is boundary (start); will be added if d has gaps
                else:
                    # consecutive day or same day: no new streak start/end decision yet
                    pass

            # compute intra-day gap if same day and we have previous timestamp
            prev_ts = last_ts.get(pid)
            if prev_ts is not None and prev_day is not None and d == prev_day:
                gap_hours = (t - prev_ts).total_seconds() / 3600.0
                if 0.0 < gap_hours < 24.0:
                    gaps_all.setdefault(pid, []).append(float(gap_hours))
                    gaps_by_day.setdefault(pid, {}).setdefault(d, []).append(float(gap_hours))

                    # boundary tagging for this day:
                    # - start day: either first ever day or a day that follows a >1 gap
                    # - end day: known via pending_end_day when a >1 gap is observed later, or at EOF
                    # We add boundary markers only for days that actually have gaps.
                    bd = boundary_days_with_gaps.setdefault(pid, set())

                    # start-of-streak day conditions:
                    if prev_day is None:
                        bd.add(d)
                    elif (d - prev_day).days > 1:
                        bd.add(d)

                    # if we already know an end-of-streak day for this pid, and it has gaps, mark it
                    end_cand = pending_end_day.get(pid)
                    if end_cand is not None:
                        # only add if that end day actually had gaps recorded
                        if end_cand in gaps_by_day.get(pid, {}):
                            bd.add(end_cand)
                        # clear; we "finalized" it
                        pending_end_day[pid] = None

            last_ts[pid] = t

            # update last_day only when we move to a new day for this pid
            # (this is important for dedup on active day logic)
            if prev_day is None or d != prev_day:
                last_day[pid] = d

    # finalize: the last active day for each pid is an end-of-streak boundary
    for pid, d_last in last_day.items():
        if pid in gaps_by_day and d_last in gaps_by_day[pid]:
            boundary_days_with_gaps.setdefault(pid, set()).add(d_last)

    # compute per-player median after IQR filtering
    records = []
    for pid, gaps in gaps_all.items():
        if not gaps:
            records.append({"Player_Profile_ID": pid, "f59_median_daily_time_off": np.nan})
            continue

        # IQR on all gaps
        q1 = float(np.percentile(gaps, 25))
        q3 = float(np.percentile(gaps, 75))
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr

        bd_days = boundary_days_with_gaps.get(pid, set())
        per_day = gaps_by_day.get(pid, {})

        kept: List[float] = []
        for day, day_gaps in per_day.items():
            if day in bd_days:
                # drop outliers on boundary days
                for g in day_gaps:
                    if low <= g <= high:
                        kept.append(g)
            else:
                kept.extend(day_gaps)

        if not kept:
            med = np.nan
        else:
            med = float(np.median(kept))

        records.append({"Player_Profile_ID": pid, "f59_median_daily_time_off": med})

    # Players with no gaps at all should be N/A per spec
    # (We only created records for players with gaps_all entries.)
    if records:
        out = pd.DataFrame.from_records(records)
    else:
        out = pd.DataFrame(columns=["Player_Profile_ID", "f59_median_daily_time_off"])

    if logger:
        logger.info(f"✅ F59 klaar: {len(out):,} spelers met >=1 intra-day gap")
        if len(out) > 0:
            na = out["f59_median_daily_time_off"].isna().sum()
            logger.info(f"   N/A (no daily time off after filtering): {na:,}")

    return out

# ------------------------------
# F60: Coefficient of variation of deposit amounts
# ------------------------------

def f60_deposit_amount_variability(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F60:
    Coefficient of variation (std/mean) of the amounts deposited.
    Positive real number (NaN if fewer than 2 deposits or mean==0).

    Notes:
      - No '+1 day' logic. Filter is [start, end).
      - Uses SUCCESSFUL DEPOSIT transactions from WOK_Player_Account_Transaction.
      - Streaming variance (Welford) per player: O(1) memory per PID.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f60_deposit_amount_variability")
        logger.info("▶ START F60: Deposit amount CV")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    tx_paths = tables.get("WOK_Player_Account_Transaction") or []
    if not tx_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f60_deposit_amount_variability"])

    # per pid: (n, mean, M2)
    stats: Dict[str, Tuple[int, float, float]] = {}

    for df in iter_csv_chunks(
        paths=tx_paths,
        usecols=[
            "Player_Profile_ID",
            "Transaction_Amount",
            "Transaction_Datetime",
            "Transaction_Type",
            "Transaction_Status",
        ],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        df = df[df["Player_Profile_ID"].notna()].copy()
        if df.empty:
            continue

        ts = pd.to_datetime(df["Transaction_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df["ts"] = ts
        df = df[df["ts"].notna()].copy()
        if df.empty:
            continue

        if start_datum is not None:
            mask = (df["ts"] >= start_datum) & (df["ts"] < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        # DEPOSIT + SUCCESSFUL
        df = df[
            df["Transaction_Type"].astype(str).eq("DEPOSIT")
            & df["Transaction_Status"].astype(str).eq("SUCCESSFUL")
        ].copy()
        if df.empty:
            continue

        amt = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
        df["amt"] = amt
        df = df[df["amt"].notna() & (df["amt"] > 0)].copy()
        if df.empty:
            continue

        pid_col = df.columns.get_loc("Player_Profile_ID")
        amt_col = df.columns.get_loc("amt")

        for i in range(len(df)):
            pid = str(df.iat[i, pid_col])
            x = float(df.iat[i, amt_col])

            n, mean, m2 = stats.get(pid, (0, 0.0, 0.0))
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            m2 += delta * delta2
            stats[pid] = (n, mean, m2)

    records = []
    for pid, (n, mean, m2) in stats.items():
        if n >= 2 and mean > 0:
            var = m2 / (n - 1)
            std = float(np.sqrt(var)) if var >= 0 else np.nan
            cv = std / mean if std == std else np.nan
        else:
            cv = np.nan

        records.append({"Player_Profile_ID": pid, "f60_deposit_amount_variability": cv})

    out = pd.DataFrame.from_records(records)

    if logger:
        logger.info(f"✅ F60 klaar: {len(out):,} spelers (met >=1 deposit gezien)")
        if len(out) > 0:
            valid = out[out["f60_deposit_amount_variability"].notna()]
            logger.info(f"   CV non-NA: {len(valid):,}")

    return out


# ------------------------------
# F61: Coefficient of variation of the quotas (odds) on which bets are placed
# ------------------------------

def f61_bet_odds_variability(
    tables: Dict[str, List[Path]],
    *,
    x_tijdspad: List[str] | None = None,
    chunksize: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    F61:
    Coefficient of variation (std/mean) of the quotas/odds on which bets are placed.
    Positive real number, N/A if no bets are placed (or fewer than 2 odds, or mean==0).

    Notes:
      - No '+1 day' logic. Filter is [start, end).
      - Reads WOK_Bet and extracts odds from Bet_Parts (e.g. Part_Odds).
      - Player is taken from Bet_Transactions (your existing helper).
      - Streaming variance (Welford) per player: O(1) memory per PID.
      - If your data uses a different odds key than "Part_Odds", adjust below.
    """
    if log_path:
        logger = _setup_feature_logger(log_path, "f61_bet_odds_variability")
        logger.info("▶ START F61: Bet odds CV")
        if x_tijdspad:
            logger.info(f"  Tijdsfiltering: {x_tijdspad[0]} - {x_tijdspad[1]} (no +1 logic)")
    else:
        logger = None

    if x_tijdspad:
        start_datum = parse_ddmmyyyy_to_timestamp(x_tijdspad[0])
        eind_datum  = parse_ddmmyyyy_to_timestamp(x_tijdspad[1])
    else:
        start_datum = None
        eind_datum  = None

    bet_paths = tables.get("WOK_Bet") or []
    if not bet_paths:
        return pd.DataFrame(columns=["Player_Profile_ID", "f61_bet_odds_variability"])

    # per pid: (n, mean, M2)
    stats: Dict[str, Tuple[int, float, float]] = {}

    # organisatie-relationeel: spelers (WOK_Bet_Transaction) + parts/odds (WOK_Bet_Parts) per bet, join op pk_id.
    _bet_tx_map = build_bet_tx_map(tables.get("WOK_Bet_Transaction"), chunksize=chunksize)
    _bet_parts_map = build_bet_parts_map(tables.get("WOK_Bet_Parts"), chunksize=chunksize)

    for df in iter_csv_chunks(
        paths=bet_paths,
        usecols=["pk_id", "Bet_Start_Datetime"],
        chunksize=chunksize,
        verbose=verbose,
    ):
        if df.empty:
            continue

        if start_datum is not None:
            ts = pd.to_datetime(df["Bet_Start_Datetime"], errors="coerce", utc=True).dt.tz_localize(None)
            mask = (ts >= start_datum) & (ts < eind_datum)
            if not mask.any():
                continue
            df = df.loc[mask].copy()

        pk_col = df.columns.get_loc("pk_id")

        for i in range(len(df)):
            pk = df.iat[i, pk_col]

            # player id (WOK_Bet_Transaction-join)
            pids = [str(p) for p, _t in _bet_tx_map.get(pk, [])
                    if p is not None and not (isinstance(p, float) and pd.isna(p))]
            if not pids:
                continue
            pid = str(pids[0])

            # odds extraction: iterate parts and read part_odds (WOK_Bet_Parts-join)
            for part_obj in _bet_parts_map.get(pk, []):
                if not isinstance(part_obj, dict):
                    continue
                v = part_obj.get("part_odds", None)
                if v is None or v == "" or (isinstance(v, float) and pd.isna(v)):
                    continue
                try:
                    x = float(v)
                except (ValueError, TypeError):
                    continue
                if x <= 0:
                    continue

                n, mean, m2 = stats.get(pid, (0, 0.0, 0.0))
                n += 1
                delta = x - mean
                mean += delta / n
                delta2 = x - mean
                m2 += delta * delta2
                stats[pid] = (n, mean, m2)

    records = []
    for pid, (n, mean, m2) in stats.items():
        if n >= 2 and mean > 0:
            var = m2 / (n - 1)
            std = float(np.sqrt(var)) if var >= 0 else np.nan
            cv = std / mean if std == std else np.nan
        else:
            cv = np.nan

        records.append({"Player_Profile_ID": pid, "f61_bet_odds_variability": cv})

    if records:
        out = pd.DataFrame.from_records(records)
    else:
        print("   ⚠️  F61: Geen betting data gevonden voor deze operator — lege DataFrame geretourneerd.")
        if logger:
            logger.warning("⚠️  F61: Geen betting data gevonden voor deze operator — lege DataFrame geretourneerd.")
        out = pd.DataFrame(columns=["Player_Profile_ID", "f61_bet_odds_variability"])

    if logger:
        logger.info(f"✅ F61 klaar: {len(out):,} spelers (met >=1 odds gezien)")
        if len(out) > 0:
            valid = out[out["f61_bet_odds_variability"].notna()]
            logger.info(f"   CV non-NA: {len(valid):,}")

    return out


























# ============================================================================
# JSON Parsing Functions - Extracted from various WOK tables
# ============================================================================
# These functions demonstrate how to parse JSON columns from different table types.
# Based on examples in feature_engineering.py

def nr_of_bank_accounts(
    tables: Dict[str, List[Path]],
    *,
    chunksize: int = 200_000,
    batch_out: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Haal alle bank IDs uit de JSON kolom Player_Profile_Bank_Account.
    Geef een dataframe terug wat per Player_Profile_ID het aantal bank accounts bevat.

    Player_Profile_Bank_Account format:
    [{"Bank_Account_ID":"13zsdfa","Bank_Account_Datetime":"2025-06-19T13:10:39Z","Bank_Account_Active":"true"},
    {"Bank_Account_ID":"21t1tb","Bank_Account_Datetime":"2025-06-20T11:05:21Z","Bank_Account_Active":"false"}]

    Output:
        DataFrame met kolommen:
        - Player_Profile_ID
        - number_of_bank_accounts
    """
    player_profile_paths = tables.get("WOK_Player_Profile")
    if player_profile_paths is None:
        raise FileNotFoundError("Need WOK_Player_Profile.")

    logger = _setup_feature_logger(
        (Path(player_profile_paths[0]).parent / "feature_nr_of_bank_accounts.log") if log_path is None else log_path,
        "nr_of_bank_accounts"
    )
    logger.info("▶ START feature 'nr_of_bank_accounts' (streaming)")
    logger.info(f"profile_files={len(player_profile_paths)} | chunksize={chunksize}")

    need_cols = ["Player_Profile_ID", "Player_Profile_Bank_Account"]
    dict_bank_ids_per_speler = {}

    for df in iter_csv_chunks(
        paths=player_profile_paths,
        usecols=need_cols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        # Bepaal kolomindexen vooraf (efficiënter)
        col_idx_id = df.columns.get_loc(need_cols[0])
        col_idx_json = df.columns.get_loc(need_cols[1])

        for rij in range(len(df)):
            personID = df.iat[rij, col_idx_id]
            Player_Profile_Bank_Account_json = df.iat[rij, col_idx_json]

            if personID not in dict_bank_ids_per_speler:
                dict_bank_ids_per_speler[personID] = set()

            bestaande_bank_accounts = dict_bank_ids_per_speler[personID]

            # Gebruik de iterator om Bank_Account_IDs uit JSON te halen
            for bank_account_ID in simple_Player_Profile_Bank_Account_json_iterator(
                json_obj=Player_Profile_Bank_Account_json,
                needed_vars=["Bank_Account_ID"]
            ):
                bestaande_bank_accounts.add(bank_account_ID)
                dict_bank_ids_per_speler[personID] = bestaande_bank_accounts

    out = pd.DataFrame({
        "Player_Profile_ID": list(dict_bank_ids_per_speler.keys()),
        "number_of_bank_accounts": list(dict_bank_ids_per_speler.values()),
    })
    out["number_of_bank_accounts"] = out["number_of_bank_accounts"].apply(
        lambda x: len(x) if isinstance(x, set) else 0
    )

    logger.info(f"✅ nr_of_bank_accounts klaar: {len(out):,} spelers")
    if len(out) > 0:
        logger.info(f"   Gemiddeld aantal bank accounts: {out['number_of_bank_accounts'].mean():.2f}")

    return out


def nr_of_risk_classes(
    tables: Dict[str, List[Path]],
    *,
    chunksize: int = 200_000,
    batch_out: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Haal alle unieke RG_Class_Value waarden uit de JSON kolom Flag_RG_Class van WOK_Player_Flags
    en tel deze per Player_Profile_ID.

    Flag_RG_Class format:
    "{\"RG_Class_Value\": \"NO_RISK_ASSIGNED\", \"RG_Class_Datetime\": \"2017-02-28T19:17:28Z\"}"

    Output:
        DataFrame met kolommen:
        - Player_Profile_ID
        - number_of_risk_classes
    """
    player_flags_paths = tables.get("WOK_Player_Flags")
    if player_flags_paths is None:
        raise FileNotFoundError("Need WOK_Player_Flags.")

    logger = _setup_feature_logger(
        (Path(player_flags_paths[0]).parent / "feature_nr_of_risk_classes.log") if log_path is None else log_path,
        "nr_of_risk_classes"
    )
    logger.info("▶ START feature 'nr_of_risk_classes' (streaming)")
    logger.info(f"flags_files={len(player_flags_paths)} | chunksize={chunksize}")

    need_cols = ["Player_Profile_ID", "Flag_RG_Class"]
    dict_flags_ids_per_speler = {}

    for df in iter_csv_chunks(
        paths=player_flags_paths,
        usecols=need_cols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        col_idx_id = df.columns.get_loc(need_cols[0])
        col_idx_json = df.columns.get_loc(need_cols[1])

        for rij in range(len(df)):
            personID = df.iat[rij, col_idx_id]
            Flag_RG_Class_json = df.iat[rij, col_idx_json]

            if personID not in dict_flags_ids_per_speler:
                dict_flags_ids_per_speler[personID] = set()

            bestaande_flags = dict_flags_ids_per_speler[personID]

            # Gebruik de iterator om RG_Class_Value uit JSON te halen
            for RG_Class_Value in simple_RG_Class_Value_from_FLAG_RG_CLASS_json_iterator(
                json_obj=Flag_RG_Class_json,
                needed_vars=["Flag_RG_Class"]
            ):
                if RG_Class_Value is None:
                    logger.debug(f"Geen RG_Class_Value voor speler {personID} in rij {rij}")
                    continue
                bestaande_flags.add(RG_Class_Value)
                dict_flags_ids_per_speler[personID] = bestaande_flags

    out = pd.DataFrame({
        "Player_Profile_ID": list(dict_flags_ids_per_speler.keys()),
        "number_of_risk_classes": list(dict_flags_ids_per_speler.values()),
    })
    out["number_of_risk_classes"] = out["number_of_risk_classes"].apply(
        lambda x: len(x) if isinstance(x, set) else 0
    )

    logger.info(f"✅ nr_of_risk_classes klaar: {len(out):,} spelers")
    if len(out) > 0:
        logger.info(f"   Gemiddeld aantal risk classes: {out['number_of_risk_classes'].mean():.2f}")

    return out


def latest_limits(
    tables: Dict[str, List[Path]],
    *,
    chunksize: int = 200_000,
    batch_out: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Haal drie waarden uit JSON kolommen van WOK_Player_Limits:
    - Deposit_Amount uit Limit_Deposit
    - Login_Time_Window uit Limit_Login
    - Balance_Amount uit Limit_Balance

    Pak voor elke Player_Profile_ID de laatste waarde (op basis van timestamp).

    Voorbeelden van input:
    "[{\"Deposit_Request_Datetime\": \"1-1-1T1:1:1Z\", \"Deposit_Start_Datetime\": \"1-1-1T1:1:1Z\", \"Deposit_Amount\": 1, \"Deposit_Time_Window\": \"Week\"}]"
    "[{\"Login_Request_Datetime\": \"1-1-1T1:1:1Z\", \"Login_Start_Datetime\": \"1-1-1T1:1:1Z\", \"Login_Duration\": 1.1, \"Login_Time_Window\": \"Week\"}]"
    "[{\"Balance_Request_Datetime\": \"1-1-1T1:1:1Z\", \"Balance_Start_Datetime\": \"1-1-1T1:1:1Z\", \"Balance_Amount\": 1}]"

    Output:
        DataFrame met kolommen:
        - Player_Profile_ID
        - latest_deposit_amount_limit
        - latest_deposit_time_window_limit
        - latest_login_duration_limit
        - latest_login_time_window_limit
        - latest_balance_amount_limit
    """
    player_limits_paths = tables.get("WOK_Player_Limits")
    if player_limits_paths is None:
        raise FileNotFoundError("Need WOK_Player_Limits.")

    logger = _setup_feature_logger(
        (Path(player_limits_paths[0]).parent / "feature_latest_limits.log") if log_path is None else log_path,
        "latest_limits"
    )
    logger.info("▶ START feature 'latest_limits' (streaming)")
    logger.info(f"limits_files={len(player_limits_paths)} | chunksize={chunksize}")

    need_cols = ["Player_Profile_ID", "Limit_Deposit", "Limit_Login", "Limit_Balance"]

    DEPOSIT_INDEX, LOGIN_INDEX, BALANCE_INDEX = 0, 1, 2
    limit_dict_per_speler: Dict[str, list] = {}

    for df in iter_csv_chunks(
        paths=player_limits_paths,
        usecols=need_cols,
        chunksize=chunksize,
        verbose=verbose
    ):
        col_idx_id  = df.columns.get_loc("Player_Profile_ID")
        col_idx_dep = df.columns.get_loc("Limit_Deposit")
        col_idx_log = df.columns.get_loc("Limit_Login")
        col_idx_bal = df.columns.get_loc("Limit_Balance")

        for rij in range(len(df)):
            personID = df.iat[rij, col_idx_id]
            if personID not in limit_dict_per_speler:
                limit_dict_per_speler[personID] = [None, None, None]

            # Parse Deposit limits
            for timestamp, amount, window in iter_limit_values(df.iat[rij, col_idx_dep], type_="deposit"):
                prev = limit_dict_per_speler[personID][DEPOSIT_INDEX]
                if timestamp and (prev is None or (prev[0] and timestamp > prev[0])):
                    limit_dict_per_speler[personID][DEPOSIT_INDEX] = (timestamp, amount, window)

            # Parse Login limits
            for timestamp, duration, window in iter_limit_values(df.iat[rij, col_idx_log], type_="login"):
                prev = limit_dict_per_speler[personID][LOGIN_INDEX]
                if timestamp and (prev is None or (prev[0] and timestamp > prev[0])):
                    limit_dict_per_speler[personID][LOGIN_INDEX] = (timestamp, duration, window)

            # Parse Balance limits
            for timestamp, amount, _ in iter_limit_values(df.iat[rij, col_idx_bal], type_="balance"):
                prev = limit_dict_per_speler[personID][BALANCE_INDEX]
                if timestamp and (prev is None or (prev[0] and timestamp > prev[0])):
                    limit_dict_per_speler[personID][BALANCE_INDEX] = (timestamp, amount, None)

    rows = []
    for personID, triple in limit_dict_per_speler.items():
        dep, logn, bal = triple
        rows.append({
            "Player_Profile_ID": personID,
            "latest_deposit_amount_limit": dep[1] if dep else None,
            "latest_deposit_time_window_limit": dep[2] if dep else None,
            "latest_login_duration_limit": logn[1] if logn else None,
            "latest_login_time_window_limit": logn[2] if logn else None,
            "latest_balance_amount_limit": bal[1] if bal else None,
        })

    out = pd.DataFrame(rows)

    logger.info(f"✅ latest_limits klaar: {len(out):,} spelers")
    if len(out) > 0:
        logger.info(f"   Spelers met deposit limit: {out['latest_deposit_amount_limit'].notna().sum():,}")
        logger.info(f"   Spelers met login limit: {out['latest_login_duration_limit'].notna().sum():,}")
        logger.info(f"   Spelers met balance limit: {out['latest_balance_amount_limit'].notna().sum():,}")

    return out


def avg_nr_of_game_transactions(
    tables: Dict[str, List[Path]],
    *,
    usecols: Optional[List[str]] = None,
    chunksize: int = 200_000,
    batch_out: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Elke game session heeft meerdere transactions in de Game_Transactions JSON kolom.

    Momenteel is het format van de kolom Game_Transactions:
    "{\"Game_Transaction\":[{\"Player_Profile_ID\":\"12345\",\"Transaction_ID\":\"tx67890\"},
    {\"Player_Profile_ID\":\"12345\",\"Transaction_ID\":\"tx67891\"}]}"

    We tellen voor elke sessie het aantal transacties en middelen dat over alle sessies
    van die speler.

    Output:
        DataFrame met kolommen:
        - Player_Profile_ID
        - avg_number_of_game_transactions
    """
    game_session_paths = tables.get("WOK_Game_Session")
    if game_session_paths is None:
        raise FileNotFoundError("Need WOK_Game_Session.")

    logger = _setup_feature_logger(
        (Path(game_session_paths[0]).parent / "feature_avg_nr_of_game_transactions.log") if log_path is None else log_path,
        "avg_nr_of_game_transactions",
    )
    logger.info("▶ START feature 'avg_nr_of_game_transactions' (streaming)")
    logger.info(f"session_files={len(game_session_paths)} | chunksize={chunksize}")

    # Accumulators per speler
    alle_transacties_per_speler: Dict[str, int] = {}
    aantal_sessies_per_speler: Dict[str, int] = {}

    for df in iter_csv_chunks(
        paths=game_session_paths,
        usecols=usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        kol_idx_json = df.columns.get_loc("Game_Transactions")

        for rij_index in range(len(df)):
            json_veld = df.iat[rij_index, kol_idx_json]

            # Tel transacties in deze sessie per speler
            aantal_in_deze_sessie = 0
            speler_id_in_sessie = None

            for speler_id, _transaction_id in iter_transaction_ids_from_Game_Transactions(json_veld):
                aantal_in_deze_sessie += 1
                speler_id_in_sessie = speler_id

                # Initialiseer als speler nog niet gezien
                if speler_id not in alle_transacties_per_speler:
                    alle_transacties_per_speler[speler_id] = 0
                    aantal_sessies_per_speler[speler_id] = 0

            # Accumuleer voor gemiddelde (tel slechts 1x per sessie)
            if speler_id_in_sessie is not None:
                alle_transacties_per_speler[speler_id_in_sessie] += aantal_in_deze_sessie
                aantal_sessies_per_speler[speler_id_in_sessie] += 1

    # Bereken gemiddelde
    rows = []
    for speler_id in alle_transacties_per_speler:
        totaal_tx = alle_transacties_per_speler[speler_id]
        aantal_sess = aantal_sessies_per_speler[speler_id]
        avg = totaal_tx / aantal_sess if aantal_sess > 0 else 0.0
        rows.append({
            "Player_Profile_ID": speler_id,
            "avg_number_of_game_transactions": avg,
        })

    out = pd.DataFrame(rows)

    logger.info(f"✅ avg_nr_of_game_transactions klaar: {len(out):,} spelers")
    if len(out) > 0:
        logger.info(f"   Gemiddeld aantal game transactions: {out['avg_number_of_game_transactions'].mean():.2f}")

    return out


def avg_nr_of_bet_parts_per_bet(
    tables: Dict[str, List[Path]],
    *,
    usecols: Optional[List[str]] = None,
    chunksize: int = 200_000,
    batch_out: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Elke WOK_Bet heeft meerdere Bet_Parts met een eigen ID.

    Momenteel is het format van de kolom Bet_Parts:
    "{\"Part\":[{\"Part_ID\":\"first\",\"Part_Event\":\"bla\",\"Part_Odds\":\"11.00\",\"Part_Sport\":\"FOOTBALL\",...},{\"Part_ID\":\"second\",...}]}"

    De Player_Profile_ID halen we uit Bet_Transactions:
    "{\"Bet_Transaction\":[{\"Player_Profile_ID\":\"291463a9-5ab0-f2d8-9c44-6c9d499aa0a0\",\"Transaction_ID\":\"20d231e6-38d1-4d0d-80c7-21186526e5e0\"}]}"

    We berekenen het gemiddeld aantal bet parts per bet voor elke speler.

    Output:
        DataFrame met kolommen:
        - Player_Profile_ID
        - avg_number_of_bet_parts
    """
    bet_paths = tables.get("WOK_Bet")
    if bet_paths is None:
        raise FileNotFoundError("Need WOK_Bet.")

    logger = _setup_feature_logger(
        (Path(bet_paths[0]).parent / "feature_avg_nr_of_bet_parts.log") if log_path is None else log_path,
        "avg_nr_of_bet_parts",
    )
    logger.info("▶ START feature 'avg_nr_of_bet_parts' (streaming)")
    logger.info(f"bet_files={len(bet_paths)} | chunksize={chunksize}")

    # Accumulators per speler
    totaal_parts_per_speler: Dict[str, int] = {}
    aantal_bets_per_speler: Dict[str, int] = {}

    for df in iter_csv_chunks(
        paths=bet_paths,
        usecols=usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        kol_idx_tx = df.columns.get_loc("Bet_Transactions")
        # Handle both "Bet_Parts" and "Bet_parts" column names
        if "Bet_Parts" in df.columns:
            kol_idx_parts = df.columns.get_loc("Bet_Parts")
        elif "Bet_parts" in df.columns:
            kol_idx_parts = df.columns.get_loc("Bet_parts")
        else:
            raise ValueError("Column 'Bet_Parts' or 'Bet_parts' not found in DataFrame")

        for rij_index in range(len(df)):
            json_veld_tx    = df.iat[rij_index, kol_idx_tx]
            json_veld_parts = df.iat[rij_index, kol_idx_parts]

            # Haal Player_Profile_ID(s) uit Bet_Transactions
            try:
                speler_ids = list(iter_player_profile_ids_from_Bet_Transactions(json_veld_tx))
            except (AttributeError, TypeError) as e:
                # Soms is de JSON structuur anders dan verwacht, sla deze rij over
                logger.debug(f"Kon Player_Profile_ID niet uit Bet_Transactions halen (rij {rij_index}): {e}")
                continue

            if not speler_ids:
                continue
            if len(speler_ids) > 1:
                logger.warning("Bet_Transactions bevat meerdere Player_Profile_IDs; eerste genomen.")
            speler_id = speler_ids[0]

            # Tel parts in deze bet
            aantal_parts_in_deze_bet = 0
            try:
                for _part_id, _part_obj in iter_part_ids_from_Bet_Parts(json_veld_parts):
                    aantal_parts_in_deze_bet += 1
            except (AttributeError, TypeError) as e:
                # Soms is de JSON structuur anders dan verwacht, sla deze rij over
                logger.debug(f"Kon Bet_Parts niet parsen (rij {rij_index}): {e}")
                continue

            # Initialiseer als speler nog niet gezien
            if speler_id not in totaal_parts_per_speler:
                totaal_parts_per_speler[speler_id] = 0
                aantal_bets_per_speler[speler_id] = 0

            # Accumuleer
            totaal_parts_per_speler[speler_id] += aantal_parts_in_deze_bet
            aantal_bets_per_speler[speler_id] += 1

    # Bereken gemiddelde
    rows = []
    for speler_id in totaal_parts_per_speler:
        totaal_parts = totaal_parts_per_speler[speler_id]
        aantal_bets = aantal_bets_per_speler[speler_id]
        avg = totaal_parts / aantal_bets if aantal_bets > 0 else 0.0
        rows.append({
            "Player_Profile_ID": speler_id,
            "avg_number_of_bet_parts": avg,
        })

    out = pd.DataFrame(rows)

    logger.info(f"✅ avg_nr_of_bet_parts_per_bet klaar: {len(out):,} spelers")
    if len(out) > 0:
        logger.info(f"   Gemiddeld aantal bet parts per bet: {out['avg_number_of_bet_parts'].mean():.2f}")

    return out


def nr_of_complaints_and_variance_in_nr_responses_per_complaint(
    tables: Dict[str, List[Path]],
    *,
    usecols: List[str] = ["Complaint_ID", "Complaint_Player_ID", "Responses"],
    chunksize: int = 200_000,
    batch_out: int = 200_000,
    log_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Elke WOK_Complaint heeft een 'Complaint_Player_ID' kolom en een 'Responses' kolom.

    De Responses data ziet er als volgt uit:
    "[{\"Response\":[{
    \"Response_ID\":\"start1234\",
    \"Response_Type\":\"Complaint Closing\",
    \"Response_Description\":\"Klantgegevens aangepast\",
    \"Response_Datetime\":\"2025-07-14T10:38:06Z\"
    }]}]"

    We tellen per speler:
    - Het aantal klachten
    - De variantie in het aantal responses per klacht

    Output:
        DataFrame met kolommen:
        - Player_Profile_ID
        - number_of_complaints
        - variance_responses_per_complaint
    """
    complaint_paths = tables.get("WOK_Complaint")
    if complaint_paths is None:
        raise FileNotFoundError("Need WOK_Complaint.")

    logger = _setup_feature_logger(
        (Path(complaint_paths[0]).parent / "feature_number_of_complaints.log") if log_path is None else log_path,
        "number_of_complaints",
    )
    logger.info("▶ START feature 'number_of_complaints' (streaming)")
    logger.info(f"complaint_files={len(complaint_paths)} | chunksize={chunksize}")

    # Accumulator per speler: dict met complaint_id -> aantal responses
    klachten_en_aantal_responses_per_speler = {}

    for df in iter_csv_chunks(
        paths=complaint_paths,
        usecols=usecols,
        chunksize=chunksize,
        verbose=verbose,
    ):
        kol_idx_player = df.columns.get_loc("Complaint_Player_ID")
        kol_idx_responses = df.columns.get_loc("Responses")
        kol_idx_complaint_id = df.columns.get_loc("Complaint_ID")

        for rij_index in range(len(df)):
            speler_id = df.iat[rij_index, kol_idx_player]
            json_veld = df.iat[rij_index, kol_idx_responses]
            complaint_id = df.iat[rij_index, kol_idx_complaint_id]

            try:
                new_list_of_response_ids = get_list_of_response_ids_from_Responses_list(json_veld)
            except (ZeroDivisionError, AttributeError, TypeError) as e:
                # Soms is de JSON structuur anders dan verwacht of bevat een fout
                logger.debug(f"Kon Responses niet parsen (rij {rij_index}, complaint {complaint_id}): {e}")
                new_list_of_response_ids = None

            if new_list_of_response_ids is None:
                if verbose:
                    logger.debug(f"Geen response_ids voor klacht {complaint_id} van speler {speler_id}")
            elif speler_id is None:
                if verbose:
                    logger.debug(f"Geen speler_id voor klacht {complaint_id} met responses {new_list_of_response_ids}")
            else:
                # Initialiseer speler als nog niet gezien
                if speler_id not in klachten_en_aantal_responses_per_speler:
                    if new_list_of_response_ids is not None:
                        initial_list = new_list_of_response_ids
                    else:
                        initial_list = []
                    klachten_en_aantal_responses_per_speler[speler_id] = {complaint_id: len(initial_list)}
                else:
                    # Voeg klacht toe aan bestaande speler
                    if complaint_id not in klachten_en_aantal_responses_per_speler[speler_id]:
                        aantal_responses = len(new_list_of_response_ids) if new_list_of_response_ids else 0
                        klachten_en_aantal_responses_per_speler[speler_id][complaint_id] = aantal_responses

    # Bereken aantal klachten en variantie in responses per klacht
    rows = []
    for speler_id, klachten_dict in klachten_en_aantal_responses_per_speler.items():
        aantal_klachten = len(klachten_dict)
        response_counts = list(klachten_dict.values())

        if len(response_counts) > 1:
            variance = np.var(response_counts)
        elif len(response_counts) == 1:
            variance = 0.0
        else:
            variance = 0.0

        rows.append({
            "Player_Profile_ID": speler_id,
            "number_of_complaints": aantal_klachten,
            "variance_responses_per_complaint": variance,
        })

    out = pd.DataFrame(rows)

    logger.info(f"✅ nr_of_complaints_and_variance klaar: {len(out):,} spelers")
    if len(out) > 0:
        logger.info(f"   Gemiddeld aantal complaints: {out['number_of_complaints'].mean():.2f}")
        logger.info(f"   Gemiddelde variance in responses: {out['variance_responses_per_complaint'].mean():.2f}")

    return out


# ============================================================================
# FEATURES_REGISTRY - Alle beschikbare Spanish features
# ============================================================================
# Dit is de "bron van waarheid" voor alle features in dit bestand.
# De runner (clean_and_parse.py) gebruikt deze registry om te weten welke
# tabellen en kolommen nodig zijn voor elke feature.

FEATURES_REGISTRY = {
    "f0_net_winloss": {
        "stream_fn": f0_net_winloss,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Amount",
                "Transaction_Datetime",  # Nodig voor tijdsfiltering
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f0_net_winloss.log",
        "kwargs": {},
    },
    "f1_active_days": {
        "stream_fn": f1_active_days,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Type",
            ],
        },
        "log_name": "f1_active_days.log",
        "kwargs": {},
    },
    "f3_total_wagered": {
        "stream_fn": f3_total_wagered,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Amount",
                "Transaction_Datetime",  # Nodig voor tijdsfiltering
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f3_total_wagered.log",
        "kwargs": {},
    },
    "f2_net_loss_per_day": {
        "stream_fn": f2_net_loss_per_day,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Amount",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f2_net_loss_per_day.log",
        "kwargs": {},
    },
    "f4_average_wager_per_day": {
        "stream_fn": f4_average_wager_per_day,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Amount",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f4_average_wager_per_day.log",
        "kwargs": {},
    },
    "f8_interactions_per_day": {
        "stream_fn": f8_interactions_per_day,
        "tables": ["WOK_Bet", "WOK_Bet_Transaction", "WOK_Game_Session",
                   "WOK_Game_Session_Transaction", "WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Bet": [
                "pk_id",
                "Bet_Start_Datetime",
                "Bet_Status",
            ],
            "WOK_Game_Session": [
                "pk_id",
                "Game_Session_Start_Datetime",
            ],
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
            "WOK_Bet_Transaction": [
                "wok_bet_pk_id",
                "player_profile_id",
            ],
            "WOK_Game_Session_Transaction": [
                "wok_game_session_pk_id",
                "player_profile_id",
            ],
        },
        "log_name": "f8_interactions_per_day.log",
        "kwargs": {},
    },
    "f11_withdrawals_per_day": {
        "stream_fn": f11_withdrawals_per_day,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f11_withdrawals_per_day.log",
        "kwargs": {},
    },
    "f12_deposits_per_day": {
        "stream_fn": f12_deposits_per_day,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f12_deposits_per_day.log",
        "kwargs": {},
    },
    "f14_active_period_span": {
        "stream_fn": f14_active_period_span,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
            ],
        },
        "log_name": "f14_active_period_span.log",
        "kwargs": {},
    },
    "f15_active_day_fraction": {
        "stream_fn": f15_active_day_fraction,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
            ],
        },
        "log_name": "f15_active_day_fraction.log",
        "kwargs": {},
    },
    # "f5_sex": {
    #     "stream_fn": f5_sex,
    #     "tables": ["WOK_Player_Profile"],
    #     "usecols": {
    #         "WOK_Player_Profile": [
    #             "Player_Profile_ID",
    #             # Note: Gender column checked dynamically (Player_Profile_Gender, Gender, Sex, Player_Gender)
    #         ],
    #     },
    #     "log_name": "f5_sex.log",
    #     "kwargs": {},
    # },
    "f6_age": {
        "stream_fn": f6_age,
        "tables": ["WOK_Player_Profile"],
        "usecols": {
            "WOK_Player_Profile": [
                "Player_Profile_ID",
                "Player_Profile_DOB",
            ],
        },
        "log_name": "f6_age.log",
        "kwargs": {},
    },
    "f7_rtp_deviation": {
        "stream_fn": f7_rtp_deviation,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Amount",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f7_rtp_deviation.log",
        "kwargs": {},
    },
    "f9_big_wins_per_day": {
        "stream_fn": f9_big_wins_per_day,
        "tables": ["WOK_Player_Account_Transaction", "WOK_Bet", "WOK_Bet_Transaction",
                   "WOK_Game_Session", "WOK_Game_Session_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_ID",
                "Transaction_Amount",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
            "WOK_Bet": [
                "pk_id",
                "Bet_Start_Datetime",
                "Bet_Status",
            ],
            "WOK_Game_Session": [
                "pk_id",
                "Game_Session_Start_Datetime",
            ],
            "WOK_Bet_Transaction": [
                "wok_bet_pk_id",
                "player_profile_id",
                "transactions_id",
            ],
            "WOK_Game_Session_Transaction": [
                "wok_game_session_pk_id",
                "player_profile_id",
                "transaction_id",
            ],
        },
        "log_name": "f9_big_wins_per_day.log",
        "kwargs": {},
    },
    "f10_canceled_withdrawals_per_day": {
        "stream_fn": f10_canceled_withdrawals_per_day,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f10_canceled_withdrawals_per_day.log",
        "kwargs": {},
    },
    "f13_canceled_deposits_per_day": {
        "stream_fn": f13_canceled_deposits_per_day,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f13_canceled_deposits_per_day.log",
        "kwargs": {},
    },
    "f24_payment_method_variety": {
        "stream_fn": f24_payment_method_variety,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Deposit_Instrument",
                "Transaction_Type",
            ],
        },
        "log_name": "f24_payment_method_variety.log",
        "kwargs": {},
    },
    "f25_voluntary_suspensions": {
        "stream_fn": f25_voluntary_suspensions,
        "tables": ["WOK_Player_Profile"],
        "usecols": {
            "WOK_Player_Profile": [
                "Player_Profile_ID",
                "Player_Profile_Status",
                "Player_Profile_Modified",
            ],
        },
        "log_name": "f25_voluntary_suspensions.log",
        "kwargs": {},
    },
    "f60_deposit_amount_variability": {
        "stream_fn": f60_deposit_amount_variability,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Amount",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f60_deposit_amount_variability.log",
        "kwargs": {},
    },
    "f16_account_age": {
        "stream_fn": f16_account_age,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f16_account_age.log",
        "kwargs": {},
    },
    "f17_number_of_bet_countries": {
        "stream_fn": f17_number_of_bet_countries,
        "tables": ["WOK_Bet", "WOK_Bet_Transaction", "WOK_Bet_Parts"],
        "usecols": {
            "WOK_Bet": [
                "pk_id",
                "Bet_Start_Datetime",
            ],
            "WOK_Bet_Transaction": [
                "wok_bet_pk_id",
                "player_profile_id",
            ],
            "WOK_Bet_Parts": [
                "wok_bet_pk_id",
                "part_event",
            ],
        },
        "log_name": "f17_number_of_bet_countries.log",
        "kwargs": {},
    },
    "f18_number_of_bet_sports": {
        "stream_fn": f18_number_of_bet_sports,
        "tables": ["WOK_Bet", "WOK_Bet_Transaction", "WOK_Bet_Parts"],
        "usecols": {
            "WOK_Bet": [
                "pk_id",
                "Bet_Start_Datetime",
            ],
            "WOK_Bet_Transaction": [
                "wok_bet_pk_id",
                "player_profile_id",
            ],
            "WOK_Bet_Parts": [
                "wok_bet_pk_id",
                "part_sport",
            ],
        },
        "log_name": "f18_number_of_bet_sports.log",
        "kwargs": {},
    },
    "f19_PROXY_FOR_nr_unique_competitions_by_max_bet_parts": {
        "stream_fn": f19_PROXY_FOR_nr_unique_competitions_by_max_bet_parts,
        "tables": ["WOK_Bet", "WOK_Bet_Transaction", "WOK_Bet_Parts"],
        "usecols": {
            "WOK_Bet": [
                "pk_id",
                "Bet_Start_Datetime",
            ],
            "WOK_Bet_Transaction": [
                "wok_bet_pk_id",
                "player_profile_id",
            ],
            "WOK_Bet_Parts": [
                "wok_bet_pk_id",
            ],
        },
        "log_name": "f19_max_bet_parts.log",
        "kwargs": {},
    },
    "f20_dutch_domestic_bets_percentage": {
        "stream_fn": f20_dutch_domestic_bets_percentage,
        "tables": ["WOK_Bet", "WOK_Bet_Transaction", "WOK_Bet_Parts"],
        "usecols": {
            "WOK_Bet": [
                "pk_id",
                "Bet_Start_Datetime",
            ],
            "WOK_Bet_Transaction": [
                "wok_bet_pk_id",
                "player_profile_id",
            ],
            "WOK_Bet_Parts": [
                "wok_bet_pk_id",
                "part_sport",
                "part_event",
            ],
        },
        "log_name": "f20_dutch_domestic_bets_pct.log",
        "kwargs": {},
    },
    "f22_limit_increases": {
        "stream_fn": f22_limit_increases,
        "tables": ["WOK_Player_Limits", "WOK_Player_Limits_Participation", "WOK_Player_Limits_Deposit", "WOK_Player_Limits_Login", "WOK_Player_Limits_Balance"],
        "usecols": {
            "WOK_Player_Limits": [
                "pk_id",
                "player_profile_id",
            ],
            "WOK_Player_Limits_Participation": [
                "wok_player_limit_pk_id",
                "participation_request_datetime",
                "participation_amount",
                "participation_time_window",
            ],
            "WOK_Player_Limits_Deposit": [
                "wok_player_limit_pk_id",
                "deposit_request_datetime",
                "deposit_amount",
                "deposit_time_window",
            ],
            "WOK_Player_Limits_Login": [
                "wok_player_limit_pk_id",
                "login_request_datetime",
                "login_duration",
                "login_time_window",
            ],
            "WOK_Player_Limits_Balance": [
                "wok_player_limit_pk_id",
                "balance_request_datetime",
                "balance_amount",
            ],
        },
        "log_name": "f22_limit_increases.log",
        "kwargs": {},
    },
    "f23_limit_decreases": {
        "stream_fn": f23_limit_decreases,
        "tables": ["WOK_Player_Limits", "WOK_Player_Limits_Participation", "WOK_Player_Limits_Deposit", "WOK_Player_Limits_Login", "WOK_Player_Limits_Balance"],
        "usecols": {
            "WOK_Player_Limits": [
                "pk_id",
                "player_profile_id",
            ],
            "WOK_Player_Limits_Participation": [
                "wok_player_limit_pk_id",
                "participation_request_datetime",
                "participation_amount",
                "participation_time_window",
            ],
            "WOK_Player_Limits_Deposit": [
                "wok_player_limit_pk_id",
                "deposit_request_datetime",
                "deposit_amount",
                "deposit_time_window",
            ],
            "WOK_Player_Limits_Login": [
                "wok_player_limit_pk_id",
                "login_request_datetime",
                "login_duration",
                "login_time_window",
            ],
            "WOK_Player_Limits_Balance": [
                "wok_player_limit_pk_id",
                "balance_request_datetime",
                "balance_amount",
            ],
        },
        "log_name": "f23_limit_decreases.log",
        "kwargs": {},
    },
    "f30_avg_interactions_per_session": {
        "stream_fn": f30_avg_interactions_per_session,
        "tables": ["WOK_Game_Session", "WOK_Game_Session_Transaction"],
        "usecols": {
            "WOK_Game_Session": [
                "pk_id",
                "Game_Session_ID",
                "Game_Session_Start_Datetime",
            ],
            "WOK_Game_Session_Transaction": [
                "wok_game_session_pk_id",
                "player_profile_id",
            ],
        },
        "log_name": "f30_avg_interactions_per_session.log",
        "kwargs": {},
    },
    "f29_sessions_other_predrawn_per_day": {
        "stream_fn": f29_sessions_other_predrawn_per_day,
        "tables": ["WOK_Game_Session", "WOK_Game_Session_Transaction", "WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Game_Session": [
                "pk_id",
                "Game_Session_ID",
                "Game_Session_Start_Datetime",
            ],
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
            "WOK_Game_Session_Transaction": [
                "wok_game_session_pk_id",
                "player_profile_id",
            ],
        },
        "log_name": "f29_sessions_other_predrawn_per_day.log",
        "kwargs": {},
    },
    "f46_median_seconds_bet_placed_to_resolved": {
        "stream_fn": f46_median_seconds_bet_placed_to_resolved,
        "tables": ["WOK_Bet", "WOK_Bet_Transaction"],
        "usecols": {
            "WOK_Bet": [
                "pk_id",
                "Bet_Start_Datetime",
                "Bet_Status",
                "Extraction_Date",
            ],
            "WOK_Bet_Transaction": [
                "wok_bet_pk_id",
                "player_profile_id",
            ],
        },
        "log_name": "f46_median_seconds_bet_placed_to_resolved.log",
        "kwargs": {},
    },
    "f47_median_seconds_session_start_to_period_end": {
        "stream_fn": f47_median_seconds_session_start_to_period_end,
        "tables": ["WOK_Game_Session", "WOK_Game_Session_Transaction"],
        "usecols": {
            "WOK_Game_Session": [
                "pk_id",
                "Game_Session_Start_Datetime",
            ],
            "WOK_Game_Session_Transaction": [
                "wok_game_session_pk_id",
                "player_profile_id",
            ],
        },
        "log_name": "f47_median_seconds_session_start_to_period_end.log",
        "kwargs": {},
    },
    "f48_percentage_bets_with_cashout": {
        "stream_fn": f48_percentage_bets_with_cashout,
        "tables": ["WOK_Bet", "WOK_Bet_Transaction", "WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Bet": [
                "pk_id",
                "Bet_ID",
                "Bet_Start_Datetime",
            ],
            "WOK_Bet_Transaction": [
                "wok_bet_pk_id",
                "player_profile_id",
                "transactions_id",
            ],
            "WOK_Player_Account_Transaction": [
                "Transaction_ID",
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f48_percentage_bets_with_cashout.log",
        "kwargs": {},
    },
    "f49_percentage_live_bets": {
        "stream_fn": f49_percentage_live_bets,
        "tables": ["WOK_Bet", "WOK_Bet_Transaction", "WOK_Bet_Parts"],
        "usecols": {
            "WOK_Bet": [
                "pk_id",
                "Bet_Start_Datetime",
            ],
            "WOK_Bet_Transaction": [
                "wok_bet_pk_id",
                "player_profile_id",
            ],
            "WOK_Bet_Parts": [
                "wok_bet_pk_id",
                "part_live",
            ],
        },
        "log_name": "f49_percentage_live_bets.log",
        "kwargs": {},
    },
    "f50_single_bet_percentage": {
        "stream_fn": f50_single_bet_percentage,
        "tables": ["WOK_Bet", "WOK_Bet_Transaction", "WOK_Bet_Parts"],
        "usecols": {
            "WOK_Bet": [
                "pk_id",
                "Bet_Start_Datetime",
            ],
            "WOK_Bet_Transaction": [
                "wok_bet_pk_id",
                "player_profile_id",
            ],
            "WOK_Bet_Parts": [
                "wok_bet_pk_id",
            ],
        },
        "log_name": "f50_single_bet_percentage.log",
        "kwargs": {},
    },
    "f61_bet_odds_variability": {
        "stream_fn": f61_bet_odds_variability,
        "tables": ["WOK_Bet", "WOK_Bet_Transaction", "WOK_Bet_Parts"],
        "usecols": {
            "WOK_Bet": [
                "pk_id",
                "Bet_Start_Datetime",
            ],
            "WOK_Bet_Transaction": [
                "wok_bet_pk_id",
                "player_profile_id",
            ],
            "WOK_Bet_Parts": [
                "wok_bet_pk_id",
                "part_odds",
            ],
        },
        "log_name": "f61_bet_odds_variability.log",
        "kwargs": {},
    },
    "f31_median_rounds_per_session": {
        "stream_fn": f31_median_rounds_per_session,
        "tables": ["WOK_Game_Session", "WOK_Game_Session_Transaction"],
        "usecols": {
            "WOK_Game_Session": [
                "pk_id",
                "Game_Session_Start_Datetime",
                "Game_Session_Rounds",
            ],
            "WOK_Game_Session_Transaction": [
                "wok_game_session_pk_id",
                "player_profile_id",
                "transaction_id",
            ],
        },
        "log_name": "f31_median_rounds_per_session.log",
        "kwargs": {},
    },
    "f32_game_types_count": {
        "stream_fn": f32_game_types_count,
        "tables": ["WOK_Game_Session", "WOK_Game_Session_Transaction", "WOK_Game"],
        "usecols": {
            "WOK_Game_Session": [
                "pk_id",
                "Game_Session_Start_Datetime",
                "Game_ID",
            ],
            "WOK_Game": [
                "Game_ID",
                "Game_Type",
            ],
            "WOK_Game_Session_Transaction": [
                "wok_game_session_pk_id",
                "player_profile_id",
            ],
        },
        "log_name": "f32_game_types_count.log",
        "kwargs": {},
    },
    "f40_products_per_active_day": {
        "stream_fn": f40_products_per_active_day,
        "tables": ["WOK_Game", "WOK_Game_Session", "WOK_Game_Session_Transaction", "WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Game": [
                "Game_ID",
                "Game_Commercial_Name",
            ],
            "WOK_Game_Session": [
                "pk_id",
                "Game_Session_Start_Datetime",
                "Game_ID",
            ],
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
            "WOK_Game_Session_Transaction": [
                "wok_game_session_pk_id",
                "player_profile_id",
            ],
        },
        "log_name": "f40_products_per_active_day.log",
        "kwargs": {},
    },
    "f41_heavy_play_hours_count": {
        "stream_fn": f41_heavy_play_hours_count,
        "tables": ["WOK_Game_Session", "WOK_Game_Session_Transaction", "WOK_Bet",
                   "WOK_Bet_Transaction", "WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Game_Session": [
                "pk_id",
                "Game_Session_Start_Datetime",
            ],
            "WOK_Bet": [
                "pk_id",
                "Bet_Start_Datetime",
            ],
            "WOK_Player_Account_Transaction": [
                "Transaction_ID",
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Status",
            ],
            "WOK_Game_Session_Transaction": [
                "wok_game_session_pk_id",
                "player_profile_id",
                "transaction_id",
            ],
            "WOK_Bet_Transaction": [
                "wok_bet_pk_id",
                "player_profile_id",
                "transactions_id",
            ],
        },
        "log_name": "f41_heavy_play_hours_count.log",
        "kwargs": {},
    },
    "f51_median_seconds_loss_to_next_bet": {
        "stream_fn": f51_median_seconds_loss_to_next_bet,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Type",
            ],
        },
        "log_name": "f51_median_seconds_loss_to_next_bet.log",
        "kwargs": {},
    },
    "f52_big_win_wager_increase_count": {
        "stream_fn": f52_big_win_wager_increase_count,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Amount",
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f52_big_win_wager_increase_count.log",
        "kwargs": {},
    },
    "f53_abs_gradient_wagered_around_median_date": {
        "stream_fn": f53_abs_gradient_wagered_around_median_date,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Amount",
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f53_abs_gradient_wagered_around_median_date.log",
        "kwargs": {},
    },
    "f26_balance_drop_frequency": {
        "stream_fn": f26_balance_drop_frequency,
        "tables": ["WOK_Player_Account_Transaction", "WOK_Player_Profile"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Amount",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
            "WOK_Player_Profile": [
                "Player_Profile_ID",
                "Player_Profile_EOD_Balance",
                "Extraction_Date",
            ],
        },
        "log_name": "f26_balance_drop_frequency.log",
        "kwargs": {},
    },
    "f27_deposits_after_balance_below_2_per_day": {
        "stream_fn": f27_deposits_after_balance_below_2_per_day,
        "tables": ["WOK_Player_Account_Transaction", "WOK_Player_Profile"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Amount",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
            "WOK_Player_Profile": [
                "Player_Profile_ID",
                "Player_Profile_EOD_Balance",
                "Extraction_Date",
            ],
        },
        "log_name": "f27_deposits_after_balance_below_2_per_day.log",
        "kwargs": {},
    },

    "f28_median_seconds_below2_to_deposit": {
        "stream_fn": f28_median_seconds_below2_to_deposit,
        "tables": ["WOK_Player_Account_Transaction", "WOK_Player_Profile"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Amount",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
            "WOK_Player_Profile": [
                "Player_Profile_ID",
                "Player_Profile_EOD_Balance",
                "Extraction_Date",
            ],
        },
        "log_name": "f28_median_seconds_below2_to_deposit.log",
        "kwargs": {},
    },
    "f42_morning_interaction_percentage": {
        "stream_fn": f42_morning_interaction_percentage,
        "tables": ["WOK_Game_Session", "WOK_Game_Session_Transaction", "WOK_Bet", "WOK_Bet_Transaction"],
        "usecols": {
            "WOK_Game_Session": [
                "pk_id",
                "Game_Session_Start_Datetime",
            ],
            "WOK_Bet": [
                "pk_id",
                "Bet_Start_Datetime",
            ],
            "WOK_Game_Session_Transaction": [
                "wok_game_session_pk_id",
                "player_profile_id",
            ],
            "WOK_Bet_Transaction": [
                "wok_bet_pk_id",
                "player_profile_id",
            ],
        },
        "log_name": "f42_morning_interaction_percentage.log",
        "kwargs": {},
    },
    "f43_evening_interaction_percentage": {
        "stream_fn": f43_evening_interaction_percentage,
        "tables": ["WOK_Game_Session", "WOK_Game_Session_Transaction", "WOK_Bet", "WOK_Bet_Transaction"],
        "usecols": {
            "WOK_Game_Session": [
                "pk_id",
                "Game_Session_Start_Datetime",
            ],
            "WOK_Bet": [
                "pk_id",
                "Bet_Start_Datetime",
            ],
            "WOK_Game_Session_Transaction": [
                "wok_game_session_pk_id",
                "player_profile_id",
            ],
            "WOK_Bet_Transaction": [
                "wok_bet_pk_id",
                "player_profile_id",
            ],
        },
        "log_name": "f43_evening_interaction_percentage.log",
        "kwargs": {},
    },
    "f44_morning_stakes_percentage": {
        "stream_fn": f44_morning_stakes_percentage,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Amount",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f44_morning_stakes_percentage.log",
        "kwargs": {},
    },


    # Batch 6: Game Segments & Temporal Patterns (CDB6 - 70% segment flags)
    "f33_f34_f35_f36_f37_f38_segments_cdb6": {
        "stream_fn": f33_f34_f35_f36_f37_f38_segments_cdb6,
        "tables": ["WOK_Bet", "WOK_Bet_Transaction", "WOK_Game", "WOK_Game_Session",
                   "WOK_Game_Session_Transaction", "WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Bet": [
                "pk_id",
                "Bet_Start_Datetime",
            ],
            "WOK_Game": [
                "Game_ID",
                "Game_Type",
            ],
            "WOK_Game_Session": [
                "pk_id",
                "Game_Session_Start_Datetime",
                "Game_ID",
            ],
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_ID",
                "Transaction_Datetime",
                "Transaction_Amount",
                "Transaction_Type",
                "Transaction_Status",
            ],
            "WOK_Bet_Transaction": [
                "wok_bet_pk_id",
                "player_profile_id",
                "transactions_id",
            ],
            "WOK_Game_Session_Transaction": [
                "wok_game_session_pk_id",
                "player_profile_id",
                "transaction_id",
            ],
        },
        "log_name": "f33_f34_f35_f36_f37_f38_segments_cdb6.log",
        "kwargs": {},
    },
    "f39_dominant_segment_share": {
        "stream_fn": f39_dominant_segment_share,
        "tables": ["WOK_Bet", "WOK_Bet_Transaction", "WOK_Game_Session", "WOK_Game_Session_Transaction"],
        "usecols": {
            "WOK_Bet": [
                "pk_id",
                "Bet_Total_Stake",
                "Bet_Start_Datetime",
            ],
            "WOK_Game_Session": [
                "pk_id",
                "Game_Session_Start_Datetime",
            ],
            "WOK_Bet_Transaction": [
                "wok_bet_pk_id",
                "player_profile_id",
            ],
            "WOK_Game_Session_Transaction": [
                "wok_game_session_pk_id",
                "player_profile_id",
            ],
        },
        "log_name": "f39_dominant_segment_share.log",
        "kwargs": {},
    },
    "f45_evening_stakes_percentage": {
        "stream_fn": f45_evening_stakes_percentage,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Amount",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f45_evening_stakes_percentage.log",
        "kwargs": {},
    },
    "f54_post_median_active_days_percentage": {
        "stream_fn": f54_post_median_active_days_percentage,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f54_post_median_active_days_percentage.log",
        "kwargs": {},
    },
    "f55_stake_variance_difference": {
        "stream_fn": f55_stake_variance_difference,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Amount",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f55_stake_variance_difference.log",
        "kwargs": {},
    },
    "f56_stake_cv_difference": {
        "stream_fn": f56_stake_cv_difference,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Amount",
                "Transaction_Datetime",
                "Transaction_Type",
                "Transaction_Status",
            ],
        },
        "log_name": "f56_stake_cv_difference.log",
        "kwargs": {},
    },
    "f57_longest_daily_streak": {
        "stream_fn": f57_longest_daily_streak,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
            ],
        },
        "log_name": "f57_longest_daily_streak.log",
        "kwargs": {},
    },
    "f58_longest_streak_ratio": {
        "stream_fn": f58_longest_streak_ratio,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
            ],
        },
        "log_name": "f58_longest_streak_ratio.log",
        "kwargs": {},
    },
    "f59_median_daily_time_off": {
        "stream_fn": f59_median_daily_time_off,
        "tables": ["WOK_Player_Account_Transaction"],
        "usecols": {
            "WOK_Player_Account_Transaction": [
                "Player_Profile_ID",
                "Transaction_Datetime",
            ],
        },
        "log_name": "f59_median_daily_time_off.log",
        "kwargs": {},
    },
}
