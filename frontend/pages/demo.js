import Image from 'next/image'
import { useState } from "react";

export default function Demo() {
    const demo_dict = [
        {id: 1, src: '/adriaen-brouwer_feeling.jpg'},
        {id: 2, src: '/adriaen-van-ostade_smoker.jpg'},
        {id: 3, src: '/albert-bloch_piping-pierrot.jpg'},
        {id: 4, src: '/chuck-close_self-portrait-2000.jpg'},
        {id: 5, src: '/martiros-saryan_still-life-1913.jpg'},
    ]

    const [img, setImg] = useState()
    const [ojb, setObj] = useState()
    const changeImg = e => setImg(e.target.value);
    const changeObj = e => setObj(e.target.value);

    return (
        <div>
            <h1 className='font-bold text-4xl text-center pb-10'>Demo</h1>

            <div>
                <ul className='grid grid-cols-5 place-items-center pb-10'>
                    {demo_dict.map((demo) => {
                        return (
                            <li>
                                <Image src={demo.src} objectFit='contain' width={180} height={180} />
                                <div className='text-center'>
                                    <input type='radio' name='radio-img' value={demo.id} className='mt-2' onChange={changeImg} />
                                </div>
                            </li>
                        )
                    })}
                </ul>
            </div>
            
            <div className='grid grid-cols-2 place-items-center pb-10 w-1/3 mx-auto'>
                <div>
                    <p className='font-bold text-2xl'>着目点を選択→</p>
                </div>

                <ObjectList img={img} />
            </div>
            
            <div>
                <p className='text-center text-2xl'>感想:</p>
            </div>

        </div>
    )
}

function ObjectList(props) {
    console.log(props.img)

    const obj_dict = [
        {id:1, obj: ['test1-1', 'test1-2', 'test1-3']},
        {id:2, obj: ['test2-1', 'test2-2', 'test2-3']},
        {id:3, obj: ['test3-1', 'test3-2', 'test3-3']},
        {id:4, obj: ['test4-1', 'test4-2', 'test4-3']},
        {id:5, obj: ['test5-1', 'test5-2', 'test5-3']},
    ]

    return (
        <div>
                <div>
                    <input type='radio' className='hidden' name='radio-object' />
                    <label className="flex flex-col w-full max-w-lg text-center border-2 rounded border-gray-900 p-2 my-1 text-xl hover:bg-blue-200">test1</label>
                </div>
        </div>
    )
}