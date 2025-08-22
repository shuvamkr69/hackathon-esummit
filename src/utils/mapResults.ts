// utils/mapResults.ts
export interface AnalysisResponse {
  face_detection: number; // percentage 0-1
  autism_percentage: number; // percentage 0-1
  gaze: string;
  emotion: string;
}

export interface MappedResult {
  label: string;
  value: string;
  tips?: string[];
}

export function mapAnalysisResults(response: AnalysisResponse): MappedResult[] {
  const results: MappedResult[] = [];

  // Handle face visibility first
  if (response.face_detection < 0.5) {
    results.push({
      label: "Face Detection",
      value: "Face not visible. Please ensure clear visibility."
    });
    return results; // stop here, don't show autism % or insights
  }

  // Face is visible → show detection %
  results.push({
    label: "Face Detection",
    value: `${(response.face_detection * 100).toFixed(1)}%`
  });

  // Show Autism percentage
  results.push({
    label: "Autism Likelihood",
    value: `${(response.autism_percentage * 100).toFixed(1)}%`
  });

  // If autism % is high, add supportive tips
  if (response.autism_percentage > 0.7) {
    results.push({
      label: "Helpful Tips",
      value: "Consider practicing the following:",
      tips: [
        "Maintain steady eye contact when communicating.",
        "Work on expressing emotions through facial expressions."
      ]
    });
  }

  // Gaze info
  results.push({
    label: "Eye Contact (Gaze)",
    value: response.gaze
  });

  // Emotion info
  results.push({
    label: "Facial Expression (Emotion)",
    value: response.emotion
  });

  return results;
}
