import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import mlflow
import torch
import lightgbm as lgbm
import numpy as np
import tqdm
from copy import deepcopy
from typing import Any, Union, Dict, Optional, Tuple, List

# local imports
from src.data import load_data, preprocess_data, PeptideDataset
from src.util.metrics import rMAE, rMSE
from src.models.mlp import MLP

# from src.models.rnn import RNN


class GenericModelHandler(object):
    """Generic Model Handler class from which frame-work specific handlers should inherit"""

    def __init__(
        self,
        model_name: str,
        data: pd.DataFrame,
        val_frac: float = 0.3,
        test_frac: float = 0.0,
        preprocess_data: bool = True,
        clean_data_q: Optional[Tuple[float]] = None,
        target: str = "iRT",
        vectorizer: Optional[Any] = None,
        logging_on: bool = True,
        seed: int = 42,
        remove_non_numeric: bool = True,
    ):
        """Initialize a GenericModelHandler object

        Args:
            model_name (str): Name of the model being handled
            data (pd.DataFrame): dataframe containing raw data
            val_frac (float, optional): fraction of validation data to use. Defaults to 0.3.
            test_frac (float, optional): fraction of test data to use if any. Defaults to 0.0.
            preprocess_data (bool, optional): Apply preprocessing? Defaults to True.
            clean_data_q (Optional[Tuple[float]], optional): Remove statistical outliers? Defaults to None.
            target (str, optional): target column name. Defaults to "iRT".
            vectorizer (Optional[Any], optional): which type of sequence vectorization to use. Defaults to None.
            logging_on (bool, optional): Turn model logging on/off. Defaults to True.
            seed (int, optional): random seed. Defaults to 42.
            remove_non_numeric (bool): Remove non-numeric columns? Defaults to True.
        """

        self.model_name = model_name
        self.data = deepcopy((data).fillna(-1))
        self.vectorizer = vectorizer

        if preprocess_data:
            self.data = self._preprocess_data()

        self.target = target
        self.logging_on = logging_on

        # train/valid split
        self.X_train, self.X_val = train_test_split(
            self.data, test_size=(val_frac + test_frac), shuffle=True, random_state=seed
        )
        if test_frac > 0:
            self.X_val, self.X_test = train_test_split(
                self.X_val, test_size=test_frac, shuffle=True, random_state=seed
            )

        # test split if desired
        if clean_data_q:
            self.X_train = self._clean_data(
                self.X_train, clean_data_q[0], clean_data_q[1]
            )

        self.y_train = self.X_train.pop(self.target)
        self.y_val = self.X_val.pop(self.target)
        self.y_test = self.X_test.pop(self.target) if test_frac > 0 else None

        # remove non-numeric columns
        if remove_non_numeric:
            for col in self.X_train.columns:
                if not any(
                    [
                        self.X_train[col].dtype == tp
                        for tp in (np.int8, np.int16, np.int64, float, bool)
                    ]
                ):
                    self.X_train.drop(columns=[col], inplace=True)
                    self.X_val.drop(columns=[col], inplace=True)
                    if test_frac > 0:
                        self.X_test.drop(columns=[col], inplace=True)

    def _preprocess_data(self) -> pd.DataFrame:
        """Preprocess the dataframe

        Returns:
            pd.DataFrame: The preprocessed data
        """
        self.data = preprocess_data(self.data, vectorizer=self.vectorizer).fillna(-1)

        return self.data

    def _clean_data(
        cls, df: pd.DataFrame, min_q: float = 0.005, max_q: float = 0.995
    ) -> pd.DataFrame:
        """Remove outliers from the data

        Args:
            min_q (float, optional): Lower quantile. Defaults to 0.005.
            max_q (float, optional): Upper quantile. Defaults to 0.995.
            target (str, optional): target column. Defaults to "iRT".

        Returns:
            pd.DataFrame: Cleaned pd.DataFrame
        """

        min_iRT, max_iRT = df["iRT"].quantile(min_q), df["iRT"].quantile(max_q)
        return df.query("iRT < @max_iRT & iRT > @min_iRT")

    def _train(self):
        raise NotImplementedError

    def _eval(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def train_eval(self):
        raise NotImplementedError


class LGBMModelHandler(GenericModelHandler):
    def __init__(
        self,
        model_name: str,
        data: pd.DataFrame,
        val_frac: float = 0.3,
        test_frac: float = 0.0,
        preprocess_data: bool = True,
        clean_data_q: Optional[Tuple[float]] = None,
        target: str = "iRT",
        vectorizer: Union[CountVectorizer, TfidfVectorizer] = CountVectorizer,
        logging_on: bool = True,
        model_parameters: Dict[str, Union[float, int]] = dict(
            n_estimators=500, max_depth=30, learning_rate=0.05
        ),
        train_args: Dict[str, Union[float, int]] = dict(
            verbose=100, early_stopping_rounds=30
        ),
        seed: int = 42,
    ):
        """Initialize a LightGBM Model Handler object

        Args:
            model_name (str): Name of the model being handled
            data (pd.DataFrame): dataframe containing raw data
            val_frac (float, optional): fraction of validation data to use. Defaults to 0.3.
            test_frac (float, optional): fraction of test data to use if any. Defaults to 0.0.
            preprocess_data (bool, optional): Apply preprocessing? Defaults to True.
            clean_data_q (Optional[Tuple[float]], optional): Remove statistical outliers? Defaults to None.
            target (str, optional): target column name. Defaults to "iRT".
            vectorizer (Optional[Any], optional): which type of sequence vectorization to use. Defaults to None.
            logging_on (bool, optional): Turn model logging on/off. Defaults to True.
            model_parameters (Dict[str, Union[float, int]], optional): arguments for boosting. Defaults to dict( n_estimators=500, max_depth=30, learning_rate=0.05 ).
            train_args (Dict[str, Union[float, int]], optional): arguments for fitting the booster. Defaults to dict( verbose=100, early_stopping_rounds=30 ).
            seed (int, optional): random seed. Defaults to 42.
        """

        super().__init__(
            model_name,
            data=data,
            val_frac=val_frac,
            test_frac=test_frac,
            preprocess_data=preprocess_data,
            clean_data_q=clean_data_q,
            target=target,
            vectorizer=vectorizer,
            logging_on=logging_on,
            seed=seed,
        )

        self.model = lgbm.LGBMRegressor(**model_parameters)
        self.train_args = train_args
        self.logger = mlflow.lightgbm

    def train_eval(self) -> Dict[str, float]:
        """Run a train + eval iteration on the data.

        Returns:
            Dict[str, float]: computed metrics.
        """

        with mlflow.start_run(run_name=self.model_name) as run:

            try:
                self.model = self.model.fit(
                    self.X_train,
                    self.y_train,
                    eval_set=(self.X_val, self.y_val),
                    eval_names=["validation"],
                    eval_metric=[rMSE, rMAE],
                    **self.train_args,
                )
                results = {
                    "rMAE": self.model.best_score_["validation"]["rMAE"],
                    "rMSE": self.model.best_score_["validation"]["rMSE"],
                    "R score": self.model.score(self.X_val, self.y_val),
                }

                if self.logger and self.logging_on:
                    self.logger.log_model(self.model, self.model_name)

                # log the metrics
                if self.logging_on:
                    for metric, value in results.items():
                        mlflow.log_metric("validation " + metric, value)
            finally:
                mlflow.end_run()

        return results

    def eval(self) -> Dict[str, float]:
        """Compute test set scores.

        Returns:
            Dict[str, float]: Test set metrics
        """

        results = dict()

        prediction = self.model.predict(self.X_test)
        results["R score"] = self.model.score(self.X_test, self.y_test)
        results["rMAE"] = rMAE(self.y_test, prediction)[1]
        results["rMSE"] = rMSE(self.y_test, prediction)[1]

        # log the metrics
        if self.logging_on:
            for metric, value in results.items():
                mlflow.log_metric("test set " + metric, value)
        return results


class TorchModelHandler(GenericModelHandler):
    def __init__(
        self,
        Model: torch.nn.Module,
        data: pd.DataFrame,
        val_frac: float = 0.3,
        test_frac: float = 0.0,
        preprocess_data: bool = True,
        tokenize: bool = False,
        clean_data_q: Optional[Tuple[float]] = None,
        target: str = "iRT",
        model_parameters: Dict[str, Union[float, int]] = dict(
            hidden_dim=128, hidden_layers=3, output_dim=1, dropout_prob=0.3
        ),
        train_args: Dict[str, Union[float, int]] = dict(
            epochs=5, batch_size=32, learning_rate=0.001, step=1000, gamma=0.5
        ),
        vectorizer: Optional[Any] = None,
        logging_on: bool = True,
        seed: int = 42,
        remove_non_numeric: bool = True,
    ):
        """Initialize a LightGBM Model Handler object

        Args:
            Model (torch.nn.Module): Model class to initialize
            data (pd.DataFrame): dataframe containing raw data
            val_frac (float, optional): fraction of validation data to use. Defaults to 0.3.
            test_frac (float, optional): fraction of test data to use if any. Defaults to 0.0.
            preprocess_data (bool, optional): Apply preprocessing? Defaults to True.
            tokenize (bool, optional): Use tokenization, e.g. for RNNs. Defaults to False.
            clean_data_q (Optional[Tuple[float]], optional): Remove statistical outliers? Defaults to None.
            target (str, optional): target column name. Defaults to "iRT".
            model_parameters (Dict[str, Union[float, int]], optional): Arguments used for model creation. Defaults to dict( hidden_dim=128, hidden_layers=3, output_dim=1, dropout_prob=0.3 ).
            train_args (Dict[str, Union[float, int]], optional): Arguments used during training. Defaults to dict( epochs=5, batch_size=32, learning_rate=0.01, step=1000, gamma=0.5 ).
            vectorizer (Optional[Any], optional): which type of sequence vectorization to use. Defaults to None.
            logging_on (bool, optional): Turn model logging on/off.. Defaults to True.
            seed (int, optional): random seed. Defaults to 42.
            remove_non_numeric (bool, optional): Use non-numerical columns (e.g. processed sequences for tokenization)? Defaults to True.
        """

        super().__init__(
            model_name=Model.name,
            data=data,
            val_frac=val_frac,
            test_frac=test_frac,
            preprocess_data=preprocess_data,
            clean_data_q=clean_data_q,
            target=target,
            vectorizer=vectorizer,
            logging_on=logging_on,
            seed=seed,
            remove_non_numeric=remove_non_numeric,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_args = train_args
        self.model_parameters = model_parameters
        self.logger = mlflow.pytorch

        # PyTorch dataloaders
        self.batch_size = train_args["batch_size"]

        # Tokenize for RNNs
        #
        if tokenize:
            self.tokenize()

            self.train_dataset = PeptideDataset(
                self.X_train,
                self.y_train.to_numpy(),
                batch_size=self.batch_size,
                scale_data=False,
            )
            self.valid_dataset = PeptideDataset(
                self.X_val,
                self.y_val.to_numpy(),
                batch_size=self.batch_size,
                scale_data=False,
            )
            if test_frac > 0.0:
                self.test_dataset = PeptideDataset(
                    self.X_test,
                    self.y_test.to_numpy(),
                    batch_size=self.batch_size,
                    scale_data=False,
                )
            model_parameters.update(
                vocab_size=model_parameters.get("vocab_size", len(self.vocabulary) + 1)
            )
        else:
            self.train_dataset = PeptideDataset(
                self.X_train.to_numpy(),
                self.y_train.to_numpy(),
                batch_size=self.batch_size,
                scale_data=True,
            )
            self.valid_dataset = PeptideDataset(
                self.X_val.to_numpy(),
                self.y_val.to_numpy(),
                batch_size=self.batch_size,
                scale_data=True,
            )
            if test_frac > 0.0:
                self.test_dataset = PeptideDataset(
                    self.X_test.to_numpy(),
                    self.y_test.to_numpy(),
                    batch_size=self.batch_size,
                    scale_data=True,
                )
            model_parameters.update(input_dim=len(self.X_train.columns))

        # Optimizers LR scheduling
        self.model = Model(**model_parameters)
        self.learning_rate = train_args["learning_rate"]
        self.optimizer = torch.optim.Adam(
            lr=self.learning_rate, params=self.model.parameters()
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, train_args["step"], gamma=train_args["gamma"]
        )

        self.loss_fn = torch.nn.SmoothL1Loss()

    def tokenize(self, padding_value=0):
        """Apply tokenization to the processed sequences"""

        self.vocabulary = (
            self.X_train["sequence_proc"].explode().unique().tolist()
        )  # Build only from training data!

        mapping = {
            amino: i + 1 for i, amino in enumerate(self.vocabulary)
        }  # 0 is pad id

        self.X_train["tokens"] = self.X_train["sequence_proc"].apply(
            lambda pep: torch.tensor([mapping[amino] for amino in pep])
        )
        self.X_val["tokens"] = self.X_val["sequence_proc"].apply(
            lambda pep: torch.tensor([mapping[amino] for amino in pep])
        )
        self.X_test["tokens"] = self.X_test["sequence_proc"].apply(
            lambda pep: torch.tensor([mapping[amino] for amino in pep])
        )

        self.X_train = torch.nn.utils.rnn.pad_sequence(
            self.X_train["tokens"].tolist(), padding_value=padding_value
        ).T
        self.X_val = torch.nn.utils.rnn.pad_sequence(
            self.X_val["tokens"].tolist(), padding_value=padding_value
        ).T
        self.X_test = torch.nn.utils.rnn.pad_sequence(
            self.X_test["tokens"].tolist(), padding_value=padding_value
        ).T

    def train(self):

        self.model.to(self.device)
        epochs = self.train_args.get("epochs", 1)
        for epoch in range(epochs):
            curr_loss = 0.0
            for i, (inp, target) in enumerate(
                tqdm.auto.tqdm(
                    self.train_dataset.dataloader(shuffle=True),
                    desc=f"Training epoch {epoch+1}",
                )
            ):

                self.optimizer.zero_grad()

                inp = (inp if inp.dtype == torch.int64 else inp.float()).to(self.device)
                target = target.reshape((target.shape[0], 1)).float().to(self.device)

                out = self.model(inp)

                loss = self.loss_fn(out, target)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # Print statistics
                curr_loss += loss.item()
                if i % 100 == 0 and i > 0:
                    print(
                        "Loss after  %5d iterations: %.3f @ LR %.2e"
                        % (
                            i,
                            curr_loss / 100 / target.shape[0],
                            self.scheduler.get_last_lr()[-1],
                        )
                    )
                    curr_loss = 0.0

    def eval(self, dataset):

        self.model.eval()
        self.model.to(self.device)

        rmae, rmse, r = [], [], []
        with torch.no_grad():
            for inp, target in tqdm.auto.tqdm(
                dataset.dataloader(shuffle=False), desc="Evaluating"
            ):

                inp = (inp if inp.dtype == torch.int64 else inp.float()).to(self.device)
                target = target.to(self.device)
                pred = self.model(inp).reshape((-1,))

                y_true, y_pred = (
                    target.detach().cpu().numpy(),
                    pred.detach().cpu().numpy(),
                )
                rmae.append(rMAE(y_true, y_pred)[1])
                rmse.append(rMSE(y_true, y_pred)[1])
                r.append(r2_score(y_true, y_pred))

        return {"rMAE": np.mean(rmae), "rMSE": np.mean(rmse), "R score": np.mean(r)}

    def train_eval(self) -> Dict[str, float]:
        """Run a train + eval iteration on the data.

        Returns:
            Dict[str, float]: computed metrics.
        """

        with mlflow.start_run(run_name=self.model_name) as run:

            try:
                self.train()

                results = self.eval(dataset=self.valid_dataset)

                if self.logger and self.logging_on:
                    self.logger.log_model(self.model, self.model_name)

                # log the metrics
                if self.logging_on:
                    for metric, value in results.items():
                        mlflow.log_metric("validation " + metric, value)
            finally:
                mlflow.end_run()

        return results

    def predict_all(self) -> List[float]:
        pred = []
        for loader in [
            self.train_dataset.dataloader(shuffle=False),
            self.valid_dataset.dataloader(shuffle=False),
            self.test_dataset.dataloader(shuffle=False),
        ]:
            for inp, tgt in loader:

                inp = (inp if inp.dtype == torch.int64 else inp.float()).to(self.device)
                out = self.model(inp)
                pred.append(
                    list(
                        zip(
                            tgt.detach().cpu().numpy(),
                            out.detach().cpu().numpy().squeeze(),
                        )
                    )
                )

        return np.concatenate(pred)

    def dump(self, path: str) -> None:
        """Dump model locally"""

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_parameters": self.model_parameters,
                "train_args": self.train_args,
            },
            path,
        )

    def load(self, path: str):
        """Load model from local path"""

        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model_parameters = checkpoint["model_parameters"]
        self.train_args = checkpoint["train_args"]
