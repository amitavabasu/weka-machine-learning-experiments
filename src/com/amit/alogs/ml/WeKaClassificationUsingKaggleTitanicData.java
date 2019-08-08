package com.amit.alogs.ml;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
//import weka.attributeSelection.AttributeSelection;

public class WeKaClassificationUsingKaggleTitanicData {
	
	public static void sop(String any){
		System.out.println(any);
	}
	
	public static void main(String[] args){
		sop("Weka loaded");
		try {
			//CSVLoader loader = new CSVLoader();
			//loader.setSource(new File("titanic_test.csv"));
			DataSource ds = new DataSource("titanic_train.arff");////new DataSource(loader);//
			Instances data = ds.getDataSet();
			sop("Total data attributes: "+data.numAttributes()+"; instances: "+data.numInstances());
//			sop(data.toString());
//			ArffSaver saver = new ArffSaver();
//			saver.setInstances(data);
//			saver.setFile(new File("titanic_test.arff"));
//			saver.writeBatch();
			
//			AttributeSelection filter = new AttributeSelection();
//			CfsSubsetEval eval = new CfsSubsetEval();
//			GreedyStepwise search = new GreedyStepwise();
//			search.setSearchBackwards(true);
//			filter.setEvaluator(eval);
//			filter.setSearch(search);
//			filter.setInputFormat(data);
//			Instances newData = Filter.useFilter(data, filter);
//			sop("Selected total data attributes: "+newData.numAttributes()+"; instances: "+newData.numInstances());
//			sop(newData.toString());
			
			data.setClassIndex(data.numAttributes()-1);
			
			weka.attributeSelection.AttributeSelection attSelect = new weka.attributeSelection.AttributeSelection();
			InfoGainAttributeEval eval = new InfoGainAttributeEval();
			Ranker search = new Ranker();
			attSelect.setEvaluator(eval);
			attSelect.setSearch(search);
			attSelect.SelectAttributes(data);
			sop(attSelect.toResultsString());
			int[] indices = attSelect.selectedAttributes();
			sop(Utils.arrayToString(indices));
			
			Classifier classifier = new J48();
			classifier.buildClassifier(data);
			DataSource testds = new DataSource("titanic_test.arff");////new DataSource(loader);//
			Instances testData = testds.getDataSet();
			testData.setClassIndex(testData.numAttributes()-1);
			sop("Total test-data attributes: "+testData.numAttributes()+"; instances: "+testData.numInstances());
			sop("Result");
			for(int i=0;i<testData.numInstances(); i++){
				Instance testInstance = testData.get(i);
				String pId = testInstance.toString(0);
				double result = classifier.classifyInstance(testInstance);
				String strResult;
				if(result==0){
					strResult = "No";
				}else if(result==1){
					strResult = "Yes";
				}else{
					strResult = "Unknown";
				}
				sop(pId+","+strResult);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
