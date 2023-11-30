import Link from "next/link";

export default function Hero() {
    return (
        <>
            <section className="relative" id="hero">
                <div class="flex justify-between w-full -z-[1px] absolute flex-row h-full left-0 lg:h-full lg:max-w-7xl lg:px-0 mx-auto px-6 right-0">
                    <div class="w-full h-full border-[#f5f5f510]/5 border-dashed border-x"></div>
                    <div class="w-full h-full border-[#f5f5f510]/5 border-dashed border-x"></div>
                    <div class="w-full h-full border-[#f5f5f510]/5 border-dashed border-x"></div>
                    <div class="w-full h-full border-[#f5f5f510]/5 border-dashed border-x"></div>
                    <div class="w-full h-full border-[#f5f5f510]/5 border-dashed border-x"></div>
                    <div class="w-full h-full border-[#f5f5f510]/5 border-dashed border-x"></div>
                </div>
                <section>
                    <svg class="blur-3xl absolute opacity-80 right-0" viewBox="0 0 400 400" fill="none" xmlns="http://www.w3.org/2000/svg" height="100%" width="50%"><g clip-path="url(#clip0_17_60)"><g filter="url(#filter0_f_17_60)"><path d="M128.6 0H0V322.2L332.5 211.5L128.6 0Z" fill="#9b49f2"></path><path d="M400 0H128.6L332.5 211.5L400 78.75V0Z" fill="#9b49f2"></path></g></g><defs><filter color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse" height="719.867" id="filter0_f_17_60" width="719.867" x="-159.933" y="-159.933"><feFlood flood-opacity="0" result="BackgroundImageFix"></feFlood><feBlend in2="BackgroundImageFix" result="shape" in="SourceGraphic" mode="normal"></feBlend><feGaussianBlur stdDeviation="79.9667" result="effect1_foregroundBlur_17_60"></feGaussianBlur></filter></defs></svg>
                    <div class="items-center relative max-w-7xl mx-auto w-full lg:px-16 md:px-12 lg:pb-24 lg:pt-44 pb-12 pt-32 px-5">
                        <div class="max-w-5xl mx-auto">
                            <div class="md:text-center text-left">
                                <div class="max-w-2xl gap-6 lg:gap-12 lg:items-end mx-auto">
                                    <div>
                                        <span class="text-lg font-semibold bg-clip-text bg-gradient-to-r font-medium from-purple-400 text-transparent to-green-500 tracking-widest uppercase via-esmerald-600">reDocs.ai</span>
                                        <h1 class="font-normal text-white mt-8 text-3xl">AI-powered tools to <strong>generate Code &amp; API documentation, Tests, and Code Refactors</strong> <u>from your source code files</u></h1>
                                    </div>
                                </div>
                                <div class="flex flex-col justify-center gap-3 lg:items-center lg:justify-center mt-12 sm:flex-row">
                                    <Link href="#features" class="items-center relative text-sm bg-gradient-to-br focus:ring-[#5b3c7d] hover:text-gray-900 dark:text-white focus:outline-none focus:ring-4 font-medium from-[#6112b5] group group-hover:from-[#5b3c7d] group-hover:to-[#5b3c7d] inline-flex justify-center overflow-hidden p-0.5 rounded-lg text-gray-900 to-[#6112b5] focus:shadow-lg mb-2 mr-2 w-fit" id="hero-landing-cta-button">
                                        <span class="flex items-center text-white bg-vulcan-900 duration-75 ease-in group-hover:bg-opacity-0 relative rounded-md transition-all py-2.5 px-5">What you get? <svg class="h-4 w-4 ml-1.5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor" stroke-width="1.5"><path d="M19.5 5.25l-7.5 7.5-7.5-7.5m15 6l-7.5 7.5-7.5-7.5" stroke-linecap="round" stroke-linejoin="round"></path></svg>
                                        </span>
                                    </Link>
                                </div>
                            </div>
                        </div>
                        <div class="items-center relative max-w-7xl mx-auto w-full border-white/10 lg:border lg:p-10 lg:rounded-5xl mt-12">
                            <div class="vsc-controller"></div>
                            <video autoplay="" class="w-full rounded-xl lg:rounded-2xl object-cover shadow-2xl shadow-black/80" muted="">
                                <source src="https://d1hvpg65d5e656.cloudfront.net/herovideo-code-codumentation.mp4" type="video/mp4" />
                            </video>
                        </div>
                    </div>
                </section>
            </section>
        </>
    )
}