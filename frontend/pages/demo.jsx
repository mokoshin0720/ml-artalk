import Image from 'next/image'
import { useState } from "react";

export default function Demo() {
    const demo_list = [
        {
            src: '/adriaen-brouwer_feeling.jpg', 
            attention_list: ['attentino1-1', 'attention1-2']
        },
        {
            src: '/adriaen-van-ostade_smoker.jpg', 
            attention_list: ['attentino2-1', 'attention2-2']
        },
        {
            src: '/albert-bloch_piping-pierrot.jpg', 
            attention_list: ['attentino3-1', 'attention3-2']
        },
        {
            src: '/chuck-close_self-portrait-2000.jpg', 
            attention_list: ['attentino4-1', 'attention4-2']
        },
        {
            src: '/martiros-saryan_still-life-1913.jpg', 
            attention_list: ['attentino5-1', 'attention5-2']
        },
    ]

    const [src, setSrc] = useState()
    const [attentionList, setAttentionList] = useState([])
    const [attention, setAttention] = useState()

    const changeSrc = e => {
        setSrc(e.target.value);
        setAttentionList(demo_list.find(value => value.src == e.target.value).attention_list)
    }

    console.log(src)
    console.log(attention)
    console.log('=============================')

    return (
        <div>
            <h1 className='font-bold text-4xl text-center pb-10'>Demo</h1>

            <div>
                <ul className='grid grid-cols-5 place-items-center pb-10'>
                    {demo_list.map((demo, idx) => {
                        return (
                            <li key={idx}>
                                <Image src={demo.src} objectFit='contain' width={180} height={180} />
                                <div className='text-center'>
                                    <input type='radio' name='radio-img' value={demo.src} className='mt-2' onChange={changeSrc} alt='tmp' />
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

                <AttentinoList attention_list={attentionList} setAttention={setAttention} />
            </div>
            
            <div>
                <p className='text-center text-2xl'>感想:</p>
            </div>

        </div>
    )
}

function AttentinoList(props) {
    const changeAttention = e => {
        props.setAttention(e.target.value)
    }

    return (
        <div>
            <ul>
            {props.attention_list.map((attention, idx) => {
                return (
                    <li key={idx}>
                        <input type='radio' name='radio-object' value={attention} onChange={changeAttention} />
                        <label>{attention}</label>
                    </li>
                )
            })}
            </ul>
        </div>
    )
}