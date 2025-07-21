#!/usr/bin/env python3
"""
Simple script to run the Question Volume Prediction analysis
"""

from question_volume_prediction import QuestionVolumePredictor

def main():
    print("🎯 Parliamentary Question Volume Prediction")
    print("=" * 50)
    
    try:
        # Initialize predictor
        predictor = QuestionVolumePredictor()
        
        # Run complete analysis
        results = predictor.run_complete_analysis()
        
        # Display results
        print("\n" + "="*50)
        print("📈 PREDICTION RESULTS")
        print("="*50)
        
        print(f"\n🏆 Best Performing Model:")
        print(f"   Model: {results['comparison'].iloc[0]['Model']}")
        print(f"   RMSE: {results['comparison'].iloc[0]['RMSE']:.2f}")
        print(f"   MAE: {results['comparison'].iloc[0]['MAE']:.2f}")
        
        print(f"\n🔮 Future Forecast (Next 30 Days):")
        forecast = results['forecast']
        print(f"   Average questions per day: {forecast['predicted_questions'].mean():.2f}")
        print(f"   Total questions predicted: {forecast['predicted_questions'].sum():.0f}")
        print(f"   Min predicted: {forecast['predicted_questions'].min():.2f}")
        print(f"   Max predicted: {forecast['predicted_questions'].max():.2f}")
        
        print(f"\n📊 All Models Performance:")
        print(results['comparison'].to_string(index=False))
        
        print(f"\n✅ Analysis complete! Check the generated plots:")
        print("   - time_series_decomposition.png")
        print("   - model_comparison.png") 
        print("   - predictions_comparison.png")
        print("   - future_forecast.png")
        
    except Exception as e:
        print(f"❌ Error running prediction: {str(e)}")
        print("Make sure you have all required dependencies installed:")
        print("pip install -r ml_requirements.txt")

if __name__ == "__main__":
    main() 