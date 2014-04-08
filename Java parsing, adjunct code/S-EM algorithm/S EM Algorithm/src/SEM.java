import net.sf.classifier4J.ClassifierException;
import net.sf.classifier4J.IClassifier;
import net.sf.classifier4J.bayesian.BayesianClassifier;
import net.sf.classifier4J.bayesian.IWordsDataSource;
import net.sf.classifier4J.bayesian.SimpleWordsDataSource;


public class SEM {

	/**
	 * @param args
	 * @throws ClassifierException 
	 */
	public static void main(String[] args) throws ClassifierException {
		
		SimpleWordsDataSource swds = new SimpleWordsDataSource();
		
		swds.addMatch("China");
		swds.addMatch("Japan");

		swds.addMatch("HK");
		swds.addMatch("NYC");
		swds.addMatch("London");
		swds.addMatch("Zurich");


		BayesianClassifier bc = new BayesianClassifier(swds);
		
//		try {
//			System.out.println(bc.classify("China", 1));
//		} catch (WordsDataSourceException e) {
//			e.printStackTrace();
//		} catch (ClassifierException e) {
//			e.printStackTrace();
//		}
//		
		IWordsDataSource wds = new SimpleWordsDataSource();
		IClassifier classifier = new BayesianClassifier(wds);
		System.out.println( "Matches = " + classifier.classify("This is a sentence") );
		
	}

}
