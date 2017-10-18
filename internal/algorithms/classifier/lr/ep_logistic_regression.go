package lr

import (
	"bufio"
	"github.com/pantsing/hector/internal/core"
	"github.com/pantsing/hector/internal/utils"
	"github.com/urfave/cli"
	"math"
	"os"
	"strconv"
	"strings"
)

func (algo *EPLogisticRegression) Command() cli.Command {
	return cli.Command{
		Name:     "ep",
		Usage:    "EPLogisticRegression",
		Category: "LR",
		Flags: []cli.Flag{
			cli.Float64Flag{
				Name: "beta",
			},
		},
	}
}

type EPLogisticRegressionParams struct {
	init_var, beta float64
}

type EPLogisticRegression struct {
	Model  map[int64]*utils.Gaussian
	params EPLogisticRegressionParams
}

func (algo *EPLogisticRegression) SaveModel(path string) {
	sb := utils.StringBuilder{}
	for f, g := range algo.Model {
		sb.Int64(f)
		sb.Write("\t")
		sb.Float(g.Mean)
		sb.Write("\t")
		sb.Float(g.Vari)
		sb.Write("\n")
	}
	sb.WriteToFile(path)
}

func (algo *EPLogisticRegression) LoadModel(path string) {
	file, _ := os.Open(path)
	defer file.Close()

	scaner := bufio.NewScanner(file)
	for scaner.Scan() {
		line := scaner.Text()
		tks := strings.Split(line, "\t")
		fid, _ := strconv.ParseInt(tks[0], 10, 64)
		mean, _ := strconv.ParseFloat(tks[1], 64)
		vari, _ := strconv.ParseFloat(tks[2], 64)
		g := utils.Gaussian{Mean: mean, Vari: vari}
		algo.Model[fid] = &g
	}
}

func (algo *EPLogisticRegression) Predict(sample *core.Sample) float64 {
	s := utils.Gaussian{Mean: 0.0, Vari: 0.0}
	for _, feature := range sample.Features {
		if feature.Value == 0.0 {
			continue
		}
		wi, ok := algo.Model[feature.Id]
		if !ok {
			wi = &(utils.Gaussian{Mean: 0.0, Vari: algo.params.init_var})
		}
		s.Mean += feature.Value * wi.Mean
		s.Vari += feature.Value * feature.Value * wi.Vari
	}

	t := s
	t.Vari += algo.params.beta
	return t.Integral(t.Mean / math.Sqrt(t.Vari))
}

func (algo *EPLogisticRegression) Init(ctx *cli.Context) {
	algo.Model = make(map[int64]*utils.Gaussian)
	algo.params.beta = ctx.Float64("beta")
	algo.params.init_var = 1.0
}

func (algo *EPLogisticRegression) Clear() {
	algo.Model = nil
	algo.Model = make(map[int64]*utils.Gaussian)
}

func (algo *EPLogisticRegression) Train(dataset *core.DataSet) {

	for _, sample := range dataset.Samples {
		s := utils.Gaussian{Mean: 0.0, Vari: 0.0}
		for _, feature := range sample.Features {
			if feature.Value == 0.0 {
				continue
			}
			wi, ok := algo.Model[feature.Id]
			if !ok {
				wi = &(utils.Gaussian{Mean: 0.0, Vari: algo.params.init_var})
				algo.Model[feature.Id] = wi
			}
			s.Mean += feature.Value * wi.Mean
			s.Vari += feature.Value * feature.Value * wi.Vari
		}

		t := s
		t.Vari += algo.params.beta

		t2 := utils.Gaussian{Mean: 0.0, Vari: 0.0}
		if sample.Label > 0.0 {
			t2.UpperTruncateGaussian(t.Mean, t.Vari, 0.0)
		} else {
			t2.LowerTruncateGaussian(t.Mean, t.Vari, 0.0)
		}
		t.MultGaussian(&t2)
		s2 := t
		s2.Vari += algo.params.beta
		s0 := s
		s.MultGaussian(&s2)

		for _, feature := range sample.Features {
			if feature.Value == 0.0 {
				continue
			}
			wi0 := utils.Gaussian{Mean: 0.0, Vari: algo.params.init_var}
			w2 := utils.Gaussian{Mean: 0.0, Vari: 0.0}
			wi, _ := algo.Model[feature.Id]
			w2.Mean = (s.Mean - (s0.Mean - wi.Mean*feature.Value)) / feature.Value
			w2.Vari = (s.Vari + (s0.Vari - wi.Vari*feature.Value*feature.Value)) / (feature.Value * feature.Value)
			wi.MultGaussian(&w2)
			wi_vari := wi.Vari
			wi_new_vari := wi_vari * wi0.Vari / (0.99*wi0.Vari + 0.01*wi.Vari)
			wi.Vari = wi_new_vari
			wi.Mean = wi.Vari * (0.99*wi.Mean/wi_vari + 0.01*wi0.Mean/wi.Vari)
			if wi.Vari < algo.params.init_var*0.01 {
				wi.Vari = algo.params.init_var * 0.01
			}
			algo.Model[feature.Id] = wi
		}
	}
}
