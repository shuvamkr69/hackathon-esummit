import { SignedIn, SignedOut } from '@clerk/nextjs';
import MainHero from '@/components/MainHero';
import KeyFeatures from '@/components/KeyFeatures';
import HowItWorks from '@/components/HowItWorks';
import CTASection from '@/components/CTASection';
import VideoAnalyzer from '@/components/VideoAnalyzer';

export default function Home() {
  return (
    <main>
      <MainHero />
      <KeyFeatures />
      <HowItWorks />
      <CTASection />
    </main>
  );
}
