import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import OneHotEncoder
from typing import Generator
from abc import ABC, abstractmethod
import pyreadstat
from typing import Callable
from sklearn.model_selection import StratifiedKFold

class Dataset(ABC):
    """
    Abstract base class for dataset preprocessing and splitting.

    This class defines the interface for dataset handling, including preprocessing
    for both LLM-based and traditional models, as well as generating splits for
    classification and imputation tasks.

    Parameters
    ----------
    path : str
        Path to the dataset file.

    Attributes
    ----------
    _df_raw : pd.DataFrame
        Raw dataframe loaded from path
    _df_llm : pd.DataFrame | None
        Cached preprocessed dataframe for LLM models
    _df_trad : pd.DataFrame | None
        Cached preprocessed dataframe for traditional models
    """

    @abstractmethod
    def __init__(self, path: str):
        self._df_raw: pd.DataFrame
        self._df_llm: pd.DataFrame | None = None
        self._df_trad: pd.DataFrame | None = None

    @abstractmethod
    def preprocess_llm(self) -> pd.DataFrame:
        """
        Preprocess the dataset for use with LLM-based models.

        Returns
        -------
        pd.DataFrame
            Preprocessed dataframe with text-friendly column names and string-type values.
        """
        print("Preprocessing for LLM-based models...")
        if self._df_llm is not None:
            return self._df_llm
        pass

    @abstractmethod
    def preprocess_trad(self) -> pd.DataFrame:
        """
        Preprocess the dataset for use with traditional statistical models.

        Returns
        -------
        pd.DataFrame
            Preprocessed dataframe with numerical and categorical encodings suitable
            for traditional ML models.
        """
        print("Preprocessing for traditional models...")
        if self._df_trad is not None:
            return self._df_trad
        pass

    def classification_splits(
        self,
        target_col: str,
        splits: BaseCrossValidator = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=24,
        ),
        one_hot: bool = True,
        train_mask: Callable | None = None,
        test_mask: Callable | None = None,
    ) -> Generator[
        tuple[
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
        ],
        None,
        None,
    ]:
        """
        Generate train-test splits for classification tasks.

        Parameters
        ----------
        target_col : str
            Name of the target column to predict.
        splits : BaseCrossValidator
            Scikit-learn cross-validation splitter.
        one_hot : bool, optional
            Whether to one-hot encode categorical variables for traditional models,
            by default True.
        train_mask : Callable | None, optional
            Optional function to filter training data based on a mask, by default None.
            The function should take a DataFrame of training data and return a boolean mask.
        test_mask : Callable | None, optional
            Optional function to filter test data based on a mask, by default None.
            The function should take a DataFrame of test data and return a boolean mask.

        Yields
        ------
        tuple[tuple, tuple]
            A nested tuple containing:
            - LLM data: (X_train, y_train, X_test, y_test)
            - Traditional data: (X_train, y_train, X_test, y_test)

            Each inner tuple contains four DataFrames for training and testing data.
            For traditional data, if one_hot=True, X_train and X_test will be sparse matrices
            from one-hot encoding.
        """
        if self._df_llm is None:
            self.preprocess_llm()
        if self._df_trad is None:
            self.preprocess_trad()

        target_col_loc = self._df_raw.columns.get_loc(target_col)

        target_col_llm = self._df_llm.columns[target_col_loc]
        target_col_trad = self._df_trad.columns[target_col_loc]

        for train_index, test_index in splits.split(
            self._df_raw.drop(columns=[target_col]), self._df_raw[target_col]
        ):
            if train_mask is not None:
                train_index = train_index[train_mask(self._df_raw.iloc[train_index])]
            if test_mask is not None:
                test_index = test_index[test_mask(self._df_raw.iloc[test_index])]

            X_llm_train = self._df_llm.iloc[train_index].drop(columns=[target_col_llm])
            y_llm_train = self._df_llm.iloc[train_index][target_col_llm]
            X_llm_test = self._df_llm.iloc[test_index].drop(columns=[target_col_llm])
            y_llm_test = self._df_llm.iloc[test_index][target_col_llm]

            X_trad_train = self._df_trad.iloc[train_index].drop(
                columns=[target_col_trad]
            )
            y_trad_train = self._df_trad.iloc[train_index][target_col_trad]
            X_trad_test = self._df_trad.iloc[test_index].drop(columns=[target_col_trad])
            y_trad_test = self._df_trad.iloc[test_index][target_col_trad]

            if one_hot:
                enc = OneHotEncoder(
                    sparse_output=True,
                    handle_unknown="ignore",  # NOTE: debatable could be set to ‘infrequent_if_exist’ as well
                ).fit(X_trad_train)
                X_trad_train = enc.transform(X_trad_train)
                X_trad_test = enc.transform(X_trad_test)

            yield (
                (
                    X_llm_train,
                    y_llm_train,
                    X_llm_test,
                    y_llm_test,
                ),
                (
                    X_trad_train,
                    y_trad_train,
                    X_trad_test,
                    y_trad_test,
                ),
            )

    def imputation_splits(self):
        raise NotImplementedError()


class DatasetArgyleANES2016(Dataset):
    """
    Dataset handler for the 2016 American National Election Study dataset used by Argyle et al. (2023).
    Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JPV20K
    """  # TODO link to the specific file used

    def __init__(self, path: str):
        super().__init__(path)

        self._df_raw = pd.read_pickle(path).rename(
            columns={"ground_truth": "vote_choice"}
        )

    def preprocess_llm(self) -> pd.DataFrame:
        super().preprocess_llm()
        self._df_llm = (
            self._df_raw.rename(
                # NOTE: colons are added for better readability in prompts
                columns={
                    "race": "Race:",
                    "discuss_politics": "Discuss Politics:",
                    "ideology": "Ideology:",
                    "party": "Party:",
                    "church_goer": "Church Goer:",
                    "age": "Age:",
                    "gender": "Gender:",
                    "political_interest": "Political Interest:",
                    "patriotism": "American Flag:",
                    "state": "State:",
                    "vote_choice": "Vote Choice:",
                }
            )
            .astype("string")
            .fillna("no answer")
        )
        return self._df_llm

    def preprocess_trad(self) -> pd.DataFrame:
        super().preprocess_trad()
        # the original data has cat/num columns already set up
        self._df_trad = self._df_raw
        return self._df_trad

class DatasetArgyleANES2016WithEmployment(DatasetArgyleANES2016):
    def __init__(self, path: str):
        super().__init__(path)

        anes_2016_employment_status_code = "V161277"
        anes_2016_path = "datasets/anes_timeseries_2016.sav"
        employment_status_var = (
            pd.read_spss(anes_2016_path, usecols=[anes_2016_employment_status_code])
            .replace(
                {
                    "1. Initial employment status: working now": "employed",
                    "5. Initial employment status: retired": "retired",
                    "6. Initial employment status: permanently disabled": "permanently disabled",
                    "4. Initial employment status: unemployed": "unemployed",
                    "7. Initial employment status: homemaker": "homemaker",
                    "2. Initial employment status: temporarily laid off": "temporarily laid off",
                    "8. Initial employment status: student": "student",
                    "-9. Refused employment status": None,
                }
            )
            .rename(columns={anes_2016_employment_status_code: "employment_status"})
        )

        self._df_raw = pd.concat([self._df_raw, employment_status_var], axis=1)

    def preprocess_llm(self) -> pd.DataFrame:
        super().preprocess_llm()
        self._df_llm.rename(columns={"employment_status": "Employment Status:"}, inplace=True)
        return self._df_llm

class DatasetGLES(Dataset):
    def __init__(self, path: str):
        super().__init__(path)

        features = {
            "Jahr": "Erhebungsjahr",
            # "Jahr": "Interview: Datum, Jahr", # for v2
            "Alter": "Alter",
            "Geschlecht": "Geschlecht",
            "Bildung": "Bildung: Schule",
            "Haushaltseinkommen": "Haushaltseinkommen",
            # "Haushaltseinkommen": "Haushaltsnettoeinkommen", # for v2
            "Erwerbstätigkeit": "Erwerbstaetigkeit: aktuell",
            "Religiosität": "Religiositaet",
            "Links-Rechts-Einstufung": "Links-Rechts-Einstufung: Ego",
            "Parteiidentifikation": "Parteiidentifikation (Version A)",
            "Parteiidentifikation Stärke": "Parteiidentifikation: Staerke",
            "Ost-West": "Ost/West (genaue Zuordnung von Berlin zu Ost- und Westdeutschland)",
            # "Bundesland": "Bundesland",
            "Zuwanderung": "Positionsissue: Zuwanderung, Ego",
            "Einkommensunterschiede verringern": "Einstellungen: Issue D, Einkommensunterschiede verringern",
            "Wahlentscheidung": "Wahlentscheidung: BTW, Zweitstimme (Version A)",
        }

        df = pd.read_spss(path)
        metadata = pyreadstat.read_sav(path)
        column_names_to_labels = metadata[1].__dict__["column_names_to_labels"]
        df.rename(columns=column_names_to_labels, inplace=True)
        # df["Interview: Datum, Jahr"] = df["Interview: Datum, Jahr"].astype(int) # for v2
        df.Erhebungsjahr = df.Erhebungsjahr.astype(int)
        df = df[df["Vorwahlbefragung"] == "nein"]  # only post-election data
        df.Alter = df.Alter.apply(lambda age: int(age) if isinstance(age, float) else age)
        self._df_all = df
        df = df[list(features.values())]
        df.columns = list(features.keys())
        df = df.astype(str)

        # reevert umlauts
        def revert_umlauts(text):
            if not isinstance(text, str):
                return text
            replacements = {
                "ae": "ä",
                "oe": "ö",
                "ue": "ü",
                "Ae": "Ä",
                "Oe": "Ö",
                "Ue": "Ü",
                "AE": "Ä",
                "OE": "Ö",
                "UE": "Ü",
            }
            for k, v in replacements.items():
                text = text.replace(k, v)
            return text

        df = df.map(revert_umlauts)

        # reduce vote classes
        df = df.replace(
            {
                "Wahlentscheidung": {
                    "DIE LINKE": "Linke",
                    "CDU/CSU": "CDU",
                    "SPD": "SPD",
                    "GRÜNE": "Grüne",
                    "FDP": "FDP",
                    "AfD": "AFD",
                    "PIRATEN": "andere Partei",
                    "NPD": "andere Partei",
                    "ungültig wählen": "keine Angabe",
                    "weiss nicht": "keine Angabe",
                    # "Mehrfachnennungen": "keine Angabe", only for v2 
                    "trifft nicht zu": "Nichtwähler",
                },
                # adapt party identification to vote class changes
                "Parteiidentifikation": {
                    "keine Partei; keiner Partei": "keine Partei",
                    "alle Parteien; alle Parteien gleich gut": "alle Parteien",
                    "DIE LINKE": "Linke",
                    "CDU/CSU": "CDU",
                    "CSU": "CDU",
                    "SPD": "SPD",
                    "GRÜNE": "Grüne",
                    "FDP": "FDP",
                    "AfD": "AFD",
                    "PIRATEN": "andere Partei",
                    "NPD": "andere Partei",
                },
            }
        )

        # drop vote choice missing values
        df = df[df["Wahlentscheidung"] != "keine Angabe"]

        self._df_raw = df

    def preprocess_llm(self) -> pd.DataFrame:
        super().preprocess_llm()
        # rename some scales for more LLM-friendly names
        self._df_llm = self._df_raw.replace(
            {
                "Bildung": {
                    "Abitur bzw. erweiterte Oberschule mit Abschluss 12. Klasse (Hochschulreife)": "Abitur",
                    "Realschulabschluss, Mittlere Reife, Fachschulreife oder Abschluss der polytechnischen Oberschule 10. Klasse": "Realschulabschluss",
                    "Hauptschulabschluss, Volksschulabschluss, Abschluss der polytechnischen Oberschule 8. oder 9. Klasse": "Hauptschulabschluss",
                    "Fachhochschulreife (Abschluss einer Fachoberschule etc.)": "Fachhochschulreife",
                },
                "Links-Rechts-Einstufung": {
                    "1 links": "sehr extrem links",
                    "2": "extrem links",
                    "3": "sehr links",
                    "4": "links",
                    "5": "eher links",
                    "6": "mittig",
                    "7": "eher rechts",
                    "8": "rechts",
                    "9": "sehr rechts",
                    "10": "extrem rechts",
                    "11 rechts": "sehr extrem rechts",
                },
                "Zuwanderung": {
                    "1 Zuzugsmöglichkeiten für Ausländer sollten erleichtert werden": "sehr extrem positiv",
                    "2": "extrem positiv",
                    "3": "sehr positiv",
                    "4": "positiv",
                    "5": "eher positiv",
                    "6": "mittig",
                    "7": "eher negativ",
                    "8": "negativ",
                    "9": "sehr negativ",
                    "10": "extrem negativ",
                    "11 Zuzugsmöglichkeiten für Ausländer sollten eingeschränkt werden": "sehr extrem negativ",
                },
            }
        ).rename(columns={column: column + ":" for column in self._df_raw.columns})
        return self._df_llm

    def preprocess_trad(self) -> pd.DataFrame:
        super().preprocess_trad()
        self._df_trad = self._df_raw
        return self._df_trad

class DatasetGLESNoPartyId(DatasetGLES):
    def __init__(self, path: str):
        super().__init__(path)
        self._df_raw = self._df_raw.drop(columns=["Parteiidentifikation"])

class DatasetGLES2017(DatasetGLES):
    """
    Dataset handler for the 2017 German Longitudinal Election Survey dataset used by Von Der Heyde et al. (2024).
    Source: https://doi.org/10.4232/1.13648
    """

    def __init__(self, path: str):
        super().__init__(path)
        self._df_raw = self._df_raw[self._df_raw["Jahr"] == "2017"]


class DatasetGLES2017NoPartyId(DatasetGLES2017):
    def __init__(self, path: str):
        super().__init__(path)
        self._df_raw = self._df_raw.drop(columns=["Parteiidentifikation"])


class DatasetGLES2017Demographics(DatasetGLES2017):
    def __init__(self, path: str):
        super().__init__(path)
        self._df_raw = self._df_raw.drop(columns=[
            # "Religiosität",
            "Links-Rechts-Einstufung",
            "Parteiidentifikation",
            "Parteiidentifikation Stärke",
            "Zuwanderung",
            "Einkommensunterschiede verringern",
            ])

class DatasetGLES2017OpenEnded(Dataset):
    """
    Dataset handler for the 2017 German Longitudinal Election Survey dataset used by Von Der Heyde et al. (2024).
    Source: https://doi.org/10.4232/1.13648
    """

    def __init__(self, path: str, path_open_ended: str, coded: bool = False):
        super().__init__(path)

        df = pd.read_spss(path)
        metadata = pyreadstat.read_sav(path)
        column_names_to_labels = metadata[1].__dict__["column_names_to_labels"]
        df.rename(columns=column_names_to_labels, inplace=True)
        df.Erhebungsjahr = df.Erhebungsjahr.astype(int)
        df = df[df["Vorwahlbefragung"] == "nein"]  # only post-election data
        df = df[df.Erhebungsjahr == 2017]
        surveys_cols = [
            "Politische Probleme: Wichtigstes, codiert, 1. Nennung",
            "Politische Probleme: Zweitwichtigstes, codiert, 1. Nennung",
            "Wahlentscheidung: BTW, Zweitstimme (Version A)",
        ]
        lfdn_2017_post = "Laufende Nummer Nachwahl-Querschnitt 2017"
        df = df[surveys_cols + [lfdn_2017_post]]
        df[lfdn_2017_post] = df[lfdn_2017_post].astype(pd.Int16Dtype())
        df_open_ended = pd.read_csv(
            path_open_ended,
            delimiter=";",
            encoding="latin-1",
        )
        df = df.merge(
            df_open_ended, how="outer", left_on=lfdn_2017_post, right_on="lfdn"
        )

        coded_features = {
            "Wichtigestes Problem": "Politische Probleme: Wichtigstes, codiert, 1. Nennung",
            "Zweitwichtigstes Problem": "Politische Probleme: Zweitwichtigstes, codiert, 1. Nennung",
        }

        uncoded_features = {
            "Wichtigestes Problem": "q3s",
            "Zweitwichtigstes Problem": "q4s",
        }

        features = (coded_features if coded else uncoded_features) | {
            "Wahlentscheidung": "Wahlentscheidung: BTW, Zweitstimme (Version A)",
        }

        df = df[features.values()]
        df.columns = features.keys()

        self._df_raw = df

    def preprocess_llm(self) -> pd.DataFrame:
        super().preprocess_llm()
        self._df_llm = (
            self._df_raw.rename(
                columns={column: column + ":" for column in self._df_raw.columns}
            )
            .astype(str)
            .fillna("Keine Antwort")
        )
        return self._df_llm

    def preprocess_trad(self) -> pd.DataFrame:
        super().preprocess_trad()
        self._df_trad = self._df_raw
        return self._df_trad
