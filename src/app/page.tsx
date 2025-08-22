import { SignedIn, SignedOut } from '@clerk/nextjs';
import Hero from '@/components/Hero';
import VideoAnalyzer from '@/components/VideoAnalyzer';

export default function Home() {
  return (
    <main>
      <SignedOut>
        <Hero />
      </SignedOut>
      <SignedIn>
        <VideoAnalyzer />
      </SignedIn>
    </main>
  );
}
