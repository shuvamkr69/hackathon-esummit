import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }

    // For now, return a mock response
    // In production, this would send the file to your ML server
    const mockResult = {
      autism_predictions: [
        { confidence: 0.65, class: 'Non-Autistic' },
        { confidence: 0.72, class: 'Non-Autistic' },
        { confidence: 0.58, class: 'Autistic' }
      ],
      emotions: ['neutral', 'happy', 'neutral', 'sad'],
      gaze: ['center', 'left', 'No face detected', 'right']
    };

    return NextResponse.json(mockResult);
  } catch (error) {
    console.error('Analysis error:', error);
    return NextResponse.json(
      { error: 'Analysis failed' },
      { status: 500 }
    );
  }
}