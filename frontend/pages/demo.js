import Image from 'next/image'

export default function Demo() {
    const demo_dict = [
        {id: 1, src: '/adriaen-brouwer_feeling.jpg'},
        {id: 2, src: '/adriaen-van-ostade_smoker.jpg'},
        {id: 3, src: '/albert-bloch_piping-pierrot.jpg'},
        {id: 4, src: '/chuck-close_self-portrait-2000.jpg'},
        {id: 5, src: '/martiros-saryan_still-life-1913.jpg'},
    ]

    return (
        <div>
            <h1 className='font-bold text-4xl text-center pb-10'>Demo</h1>

            <div>
                <ul className='grid grid-cols-5 place-items-center pb-10'>
                    {demo_dict.map((demo) => {
                        return (
                            <li>
                                <Image src={demo.src} objectFit='contain' width={180} height={180} />
                            </li>
                        )
                    })}
                </ul>
            </div>
            
            <div className='grid grid-cols-2 place-items-center pb-10 w-1/3 mx-auto'>
                <div>
                    <p className='font-bold text-2xl'>着目点を選択→</p>
                </div>

                <div>
                    <div>
                        <input type='radio' className='hidden' />
                        <label className="flex flex-col w-full max-w-lg text-center border-2 rounded border-gray-900 p-2 my-1 text-xl hover:bg-blue-200">test1</label>
                    </div>
                    <div>
                        <input type='radio' className='hidden' />
                        <label className="flex flex-col w-full max-w-lg text-center border-2 rounded border-gray-900 p-2 my-1 text-xl hover:bg-blue-200">test2</label>
                    </div>
                    <div>
                        <input type='radio' className='hidden' />
                        <label className="flex flex-col w-full max-w-lg text-center border-2 rounded border-gray-900 p-2 my-1 text-xl hover:bg-blue-200">test3</label>
                    </div>
                </div>
            </div>
            
            <div>
                <p className='text-center text-2xl'>感想:</p>
            </div>

        </div>
    )
}