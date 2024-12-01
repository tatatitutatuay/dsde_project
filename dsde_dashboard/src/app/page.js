'use client';

import { Button, Field, Textarea } from '@headlessui/react';
import clsx from 'clsx';
import { useState } from 'react';

import { TypewriterEffect } from '@/components/ui/typewriter-effect';

export default function Home() {
    // State for the textarea value
    const [abstract, setAbstract] = useState('');

    return (
        <main className="container mx-auto my-24 flex justify-center items-center">
            <div className="w-full flex flex-col gap-12 items-center justify-center">
                {/* Title */}
                <div className="text-center flex flex-col gap-4">
                    <TypewriterEffect
                        words={[
                            { text: 'Drop' },
                            { text: 'Your' },
                            { text: 'Abstract', className: 'text-pink-500' },
                            { text: 'Here' },
                        ]}
                        cursorClassName="bg-black/80"
                    />
                    <p className="text-black/50 text-md text-pretty">
                        This tool will generate keywords for your paper using
                        data from Chulalongkorn University Library.
                    </p>
                </div>

                {/* Textarea */}
                <div className="flex flex-col justify-center items-end gap-4 max-w-2xl w-full">
                    <TextField
                        value={abstract}
                        onChange={(e) => {
                            setAbstract(e.target.value);
                        }}
                        placeholder={'Paste your abstract here...'}
                    />
                    <Button2>Submit</Button2>
                </div>
            </div>
        </main>
    );
}

// TextField component
const TextField = ({ value, onChange, placeholder }) => {
    return (
        <div className="w-full max-w-2xl">
            <Field>
                <Textarea
                    className={clsx(
                        'mt-3 block w-full resize-none rounded-lg border-none bg-black/5 py-1.5 px-3 text-sm/6 text-black',
                        'focus:outline-none data-[focus]:outline-2 data-[focus]:-outline-offset-2 data-[focus]:outline-black/25'
                    )}
                    rows={5}
                    value={value}
                    onChange={onChange}
                    placeholder={placeholder}
                />
            </Field>
        </div>
    );
};

// Button component
const Button2 = ({ children }) => {
    return (
        <Button className="inline-flex justify-center items-center gap-2 rounded-md bg-gray-900 py-1.5 px-3 text-sm/6 font-medium text-white max-w-36 text-center shadow-inner shadow-white/10 focus:outline-none data-[hover]:bg-gray-600 data-[open]:bg-gray-700 data-[focus]:outline-1 data-[focus]:outline-white">
            {children}
        </Button>
    );
};
