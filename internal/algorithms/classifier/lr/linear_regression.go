package lr

import (
	"bufio"
	"github.com/pantsing/hector/internal/core"
	"github.com/pantsing/hector/internal/utils"
	"github.com/urfave/cli"
	"os"
	"strconv"
	"strings"
)

func (algo *LinearRegression) Command() cli.Command {
	return cli.Command{
		Name:     "linearRegr",
		Usage:    "Linear Regression",
		Category: "LR",
		Flags: []cli.Flag{
			cli.Float64Flag{
				Name: "learning-rate,lrate",
			},
			cli.Float64Flag{
				Name: "regularization,r",
			},
		},
	}
}

type LinearRegression struct {
	Model  map[int64]float64
	Params LogisticRegressionParams
}

func (algo *LinearRegression) SaveModel(path string) {
	sb := utils.StringBuilder{}
	for f, g := range algo.Model {
		sb.Int64(f)
		sb.Write("\t")
		sb.Float(g)
		sb.Write("\n")
	}
	sb.WriteToFile(path)
}

func (algo *LinearRegression) LoadModel(path string) {
	file, _ := os.Open(path)
	defer file.Close()

	scaner := bufio.NewScanner(file)
	for scaner.Scan() {
		line := scaner.Text()
		tks := strings.Split(line, "\t")
		fid, _ := strconv.ParseInt(tks[0], 10, 64)
		fw, _ := strconv.ParseFloat(tks[1], 64)
		algo.Model[fid] = fw
	}
}

func (algo *LinearRegression) Init(ctx *cli.Context) {
	algo.Model = make(map[int64]float64)
	algo.Params.LearningRate = ctx.Float64("learning-rate")
	algo.Params.Regularization = ctx.Float64("regularization")
}

func (algo *LinearRegression) Train(dataset *core.DataSet) {
	algo.Model = make(map[int64]float64)
	for step := 0; step < algo.Params.Steps; step++ {
		for _, sample := range dataset.Samples {
			prediction := algo.Predict(sample)
			err := sample.LabelDoubleValue() - prediction
			for _, feature := range sample.Features {
				model_feature_value, ok := algo.Model[feature.Id]
				if !ok {
					model_feature_value = 0.0
				}
				model_feature_value += algo.Params.LearningRate * (err*feature.Value - algo.Params.Regularization*model_feature_value)
				algo.Model[feature.Id] = model_feature_value
			}
		}
		algo.Params.LearningRate *= 0.9
	}
}

func (algo *LinearRegression) Predict(sample *core.Sample) float64 {
	ret := 0.0
	for _, feature := range sample.Features {
		model_feature_value, ok := algo.Model[feature.Id]
		if ok {
			ret += model_feature_value * feature.Value
		}
	}
	return ret
}
