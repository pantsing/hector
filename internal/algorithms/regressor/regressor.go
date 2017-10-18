package regressor

import (
	"github.com/pantsing/hector/internal/algorithms/eval"
	"github.com/pantsing/hector/internal/algorithms/internal"
	"github.com/pantsing/hector/internal/algorithms/regressor/gp"
	"github.com/pantsing/hector/internal/core"
	"github.com/pantsing/log"
	"github.com/urfave/cli"
	"math/rand"
	"os"
	"strconv"
	"time"
)

func Commands() []cli.Command {
	cmds := make([]cli.Command, 0, 1 << 5)
	for _, alog := range regressorIndex {
		cmd := alog.Command()
		if _, ok := internal.AlogCmdsChecker[cmd.Name]; ok {
			continue
		}
		internal.AlogCmdsChecker[cmd.Name] = struct{}{}
		cmd.Action = RegAlgorithmRun
		cmds = append(cmds, cmd)
	}
	return cmds
}

type Regressor interface {
	internal.Algorithm
	//Train model on a given dataset
	Train(dataset *core.RealDataSet)
	//Predict the output of an input sample
	Predict(sample *core.RealSample) float64
	SaveModel(path string)
	LoadModel(path string)
}

var regressorIndex map[string]Regressor = map[string]Regressor{
	"gp": new(gp.GaussianProcess),
}

func GetRegressor(method string) Regressor {
	rand.Seed(time.Now().UTC().UnixNano())
	return regressorIndex[method]
}

/* Regression */
func RegAlgorithmRun(ctx *cli.Context) (err error) {
	alogName := ctx.Command.Name
	regressor := GetRegressor(alogName)

	trainSetPath := ctx.String("trainSet")
	testSetPath := ctx.String("testSet")
	predictResultPath := ctx.String("predict")
	modelPath := ctx.String("model")
	global := ctx.Int64("global")
	cv := ctx.Int("cv")

	regressor.Init(ctx)

	var trainSet *core.RealDataSet
	if trainSetPath != "" {
		trainSet = core.NewRealDataSet()
		err = trainSet.Load(trainSetPath, global)
		if err != nil {
			return
		}
	}

	var testSet *core.RealDataSet
	if testSetPath != "" {
		testSet = core.NewRealDataSet()
		err = testSet.Load(testSetPath, global)
		if err != nil {
			return
		}
	}

	_, err = os.Stat(modelPath)
	existModel := !os.IsNotExist(err)
	if existModel {
		return err
	}
	regressor.LoadModel(modelPath)

	var predictions []*eval.RealPrediction
	var rmse float64
	if cv <= 1 {
		rmse, predictions = RegAlgorithmRunOnDataSet(regressor, trainSet, testSet)
		if predictions != nil {
			log.Infof("RMSE: %.20g\n", rmse)
		}
	} else {
		for part := 0; part < cv; part++ {
			cvTrainSet, cvTestSet := testSet.CVSplit(cv, part)
			rmse, predictions = RegAlgorithmRunOnDataSet(regressor, cvTrainSet, cvTestSet)
			log.Infof("RMSE: %.20g\n", rmse)
			regressor.Clear()
		}
	}

	if trainSet != nil && !existModel {
		regressor.SaveModel(modelPath)
	}

	if predictResultPath != "" {
		predictResultFile, err := os.Create(predictResultPath)
		if err != nil {
			return err
		}
		defer predictResultFile.Close()

		for i := range predictions {
			predictResultFile.WriteString(strconv.FormatFloat(predictions[i].Prediction, 'g', 5, 64) + "\n")
		}
	}
	return
}

func RegAlgorithmRunOnDataSet(regressor Regressor, trainSet, testSet *core.RealDataSet) (float64, []*eval.RealPrediction) {
	if trainSet != nil {
		regressor.Train(trainSet)
	}
	if testSet == nil {
		return 0, nil
	}
	predictions := []*eval.RealPrediction{}
	for _, sample := range testSet.Samples {
		prediction := regressor.Predict(sample)
		predictions = append(predictions, &eval.RealPrediction{Value: sample.Value, Prediction: prediction})
	}
	rmse := eval.RegRMSE(predictions)
	return rmse, predictions
}
