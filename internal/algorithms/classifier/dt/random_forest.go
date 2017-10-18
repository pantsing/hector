package dt

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"

	"github.com/pantsing/hector/internal/core"
	"github.com/urfave/cli"
)

type RandomForestParams struct {
	TreeCount    int
	FeatureCount float64
}

type RandomForest struct {
	trees               []*Tree
	params              RandomForestParams
	cart                CART
	continuous_features bool
}

func (self *RandomForest) SaveModel(path string) {
	file, _ := os.Create(path)
	defer file.Close()
	for _, tree := range self.trees {
		buf := tree.ToString()
		file.Write(buf)
		file.WriteString("\n#\n")
	}
}

func (self *RandomForest) LoadModel(path string) {
	file, _ := os.Open(path)
	defer file.Close()

	self.trees = []*Tree{}
	reader := bufio.NewReader(file)
	text := []string{}
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		line = strings.TrimSpace(line)
		if line == "#" {
			tree := Tree{}
			tree.fromString(text)
			self.trees = append(self.trees, &tree)
			text = []string{}
		} else {
			text = append(text, line)
		}
	}
	log.Println("rf tree count :", len(self.trees))
}

func (dt *RandomForest) Command() cli.Command {
	return cli.Command{
		Name:     "rf",
		Usage:    "RandomForest",
		Category: "DT",
		Flags: []cli.Flag{
			cli.IntFlag{
				Name: "tree-count,tc",
			},
			cli.Float64Flag{
				Name: "feature-count,fc",
			},
		},
	}
}

func (dt *RandomForest) Init(ctx *cli.Context) {
	dt.trees = []*Tree{}
	dt.cart.Init(ctx)
	dt.params.TreeCount = ctx.Int("tree-count")
	dt.params.FeatureCount = ctx.Float64("feature-count")
}

func (rdt *RandomForest) Clear() {
}

func (dt *RandomForest) Train(dataset *core.DataSet) {
	samples := []*core.MapBasedSample{}
	feature_weights := make(map[int64]float64)
	for _, sample := range dataset.Samples {
		if !dt.continuous_features {
			for _, f := range sample.Features {
				_, ok := feature_weights[f.Id]
				if !ok {
					feature_weights[f.Id] = f.Value
				}
				if feature_weights[f.Id] != f.Value {
					dt.continuous_features = true
				}
			}
		}
		msample := sample.ToMapBasedSample()
		samples = append(samples, msample)
	}
	dt.cart.continuous_features = dt.continuous_features

	trees := make(chan *Tree, dt.params.TreeCount)
	var wait sync.WaitGroup
	wait.Add(dt.params.TreeCount)

	for i := 0; i < dt.params.TreeCount; i++ {

		go func() {
			tree := dt.cart.SingleTreeBuild(samples, dt.params.FeatureCount, true)
			trees <- &tree
			fmt.Printf(".")
			wait.Done()
		}()
	}
	wait.Wait()
	fmt.Println()
	close(trees)
	for tree := range trees {
		dt.trees = append(dt.trees, tree)
	}
}

func (dt *RandomForest) Predict(sample *core.Sample) float64 {
	msample := sample.ToMapBasedSample()
	predictions := 0.0
	total := 0.0
	for _, tree := range dt.trees {
		node, _ := PredictBySingleTree(tree, msample)
		predictions += node.prediction.GetValue(1)
		total += 1.0
	}
	return predictions / total
}

func (dt *RandomForest) PredictMultiClass(sample *core.Sample) *core.ArrayVector {
	msample := sample.ToMapBasedSample()
	predictions := core.NewArrayVector()
	total := 0.0
	for _, tree := range dt.trees {
		node, _ := PredictBySingleTree(tree, msample)
		predictions.AddVector(node.prediction, 1.0)
		total += 1.0
	}
	predictions.Scale(1.0 / total)
	return predictions
}
