import localFont from 'next/font/local';
import './globals.css';
import Navbar from '@/components/Navbar';

const geistSans = localFont({
    src: './fonts/GeistVF.woff',
    variable: '--font-geist-sans',
    weight: '100 900',
});
const geistMono = localFont({
    src: './fonts/GeistMonoVF.woff',
    variable: '--font-geist-mono',
    weight: '100 900',
});

export const metadata = {
    title: 'A2K | Abstract2Keywords',
    description: 'As part of the DSDE project, we are developing a tool to extract keywords from scientific abstracts.',
};

export default function RootLayout({ children }) {
    return (
        <html lang="en">
            <body
                className={`${geistSans.variable} ${geistMono.variable} antialiased`}
            >
                <Navbar />
                {children}
            </body>
        </html>
    );
}
