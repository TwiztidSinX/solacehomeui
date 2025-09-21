export interface Message {
  sender: string;
  message: string;
  type?: 'ai' | 'user' | 'system' | 'error' | 'info';
  imageB64?: string | null;
  isThinking?: boolean;
  thought?: string;
  iframeUrl?: string;
  youtubeVideoId?: string;
  imageGalleryUrls?: string[];
}